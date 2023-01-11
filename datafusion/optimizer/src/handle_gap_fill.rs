//! An optimizer rule that enables gap-filling semantics,
//! using the SQL function `DATE_BIN_GAPFILL()`.

use crate::{optimizer::ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    expr_visitor::{ExprVisitable, ExpressionVisitor, Recursion},
    Aggregate, BuiltinScalarFunction,
    BuiltinScalarFunction::{DateBin, DateBinGapfill},
    Expr, Extension, LogicalPlan, Sort, UserDefinedLogicalNode,
};
use std::sync::Arc;

/// This optimizer rule enables gep-filling semantics for SQL queries
/// that contain calls to `DATE_BIN_GAPFILL()`.
///
/// In SQL a typical gap-filling query might look like this:
/// ```sql
/// SELECT
///   location,
///   DATE_BIN_GAPFILL(INTERVAL '1 minute', time, '1970-01-01T00:00:00Z') AS minute,
///   AVG(temp)
/// FROM temps
/// WHERE time > NOW() - INTERVAL '6 hours' AND time < NOW()
/// GROUP BY LOCATION, MINUTE
/// ```
/// The aggregateion step of the initial logical plan looks like this:
/// ```text
///   Aggregate: groupBy=[[datebingapfill(IntervalDayTime("60000"), temps.time, TimestampNanosecond(0, None)))]], aggr=[[AVG(temps.temp)]]
/// ```
/// However, `DATE_BIN_GAPFILL()` does not have an actual implementation like other functions.
/// Instead, the plan is transformed to this:
/// ```text
/// GapFill: groupBy=[[datebingapfill(IntervalDayTime("60000"), temps.time, TimestampNanosecond(0, None)))]], aggr=[[AVG(temps.temp)]], start=..., stop=...
///   Sort: datebingapfill(IntervalDayTime("60000"), temps.time, TimestampNanosecond(0, None))
///     Aggregate: groupBy=[[datebingapfill(IntervalDayTime("60000"), temps.time, TimestampNanosecond(0, None)))]], aggr=[[AVG(temps.temp)]]
/// ```
/// This optimizer rule makes that transformation.
pub struct HandleGapFill;

impl HandleGapFill {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for HandleGapFill {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizerRule for HandleGapFill {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        handle_gap_fill(plan)
    }

    fn name(&self) -> &str {
        "handle_gap_fill"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }
}

fn handle_gap_fill(plan: &LogicalPlan) -> Result<Option<LogicalPlan>> {
    let res = match plan {
        LogicalPlan::Aggregate(_) => handle_aggregate(plan)?,
        _ => None,
    };

    if res.is_none() {
        // no transformation was applied,
        // so make sure the plan is not using gap filling
        // functions in an unsupported way.
        check_node(plan)?;
    }

    Ok(res)
}

fn handle_aggregate(plan: &LogicalPlan) -> Result<Option<LogicalPlan>> {
    let (group_expr, aggr_expr, input) = match &plan {
        LogicalPlan::Aggregate(Aggregate {
            group_expr,
            aggr_expr,
            input,
            ..
        }) => (group_expr, aggr_expr, input),
        _ => unreachable!(),
    };

    let new_group_expr = replace_date_bin_gapfill(group_expr)?;
    if new_group_expr.is_none() {
        // nothing to do
        return Ok(None);
    }
    // new_group_expr has DATE_BIN_GAPFILL replaced with DATE_BIN.
    let (new_group_expr, dbg_idx) = new_group_expr.unwrap();

    let new_aggr_plan = {
        let new_aggr_plan =
            Aggregate::try_new(input.clone(), new_group_expr, aggr_expr.clone())?;
        let new_aggr_plan = LogicalPlan::Aggregate(new_aggr_plan);
        check_node(&new_aggr_plan)?;
        new_aggr_plan
    };

    let new_sort_plan = {
        let mut sort_exprs: Vec<_> = new_aggr_plan
            .schema()
            .fields()
            .iter()
            .take(group_expr.len())
            .map(|f| Expr::Column(f.qualified_column()))
            .collect();
        // ensure that date_bin_gapfill is the last sort expression.
        let last_elm = sort_exprs.len() - 1;
        sort_exprs.swap(dbg_idx, last_elm);

        LogicalPlan::Sort(Sort {
            expr: sort_exprs,
            input: Arc::new(new_aggr_plan),
            fetch: None,
        })
    };

    let new_gap_fill_plan = {
        let mut new_group_expr: Vec<_> = new_sort_plan
            .schema()
            .fields()
            .iter()
            .map(|f| Expr::Column(f.qualified_column()))
            .collect();
        let aggr_expr = new_group_expr.split_off(group_expr.len());
        LogicalPlan::Extension(Extension {
            node: Arc::new(GapFill {
                inner: LogicalPlan::Aggregate(Aggregate::try_new(
                    Arc::new(new_sort_plan),
                    new_group_expr,
                    aggr_expr,
                )?),
            }),
        })
    };

    Ok(Some(new_gap_fill_plan))
}

// Iterate over the group expression list.
// If it finds no occurrences of date_bin_gapfill at the top of
// each expression tree, it will return None.
// If it finds such an occurrence, it will return a new expression list
// with the date_bin_gapfill replaced with date_bin, and the index of
// where the replacement occurred.
fn replace_date_bin_gapfill(
    group_expr: &Vec<Expr>,
) -> Result<Option<(Vec<Expr>, usize)>> {
    let has_date_bin_gapfill = group_expr.iter().any(|e| {
        matches!(
            e,
            Expr::ScalarFunction {
                fun: DateBinGapfill,
                ..
            }
        )
    });
    if !has_date_bin_gapfill {
        return Ok(None);
    }

    let mut new_aggr_expr = Vec::with_capacity(group_expr.len());
    let mut dbg_idx = None;

    group_expr
        .iter()
        .enumerate()
        .try_for_each(|(i, e)| -> Result<()> {
            let new_expr = match e {
                Expr::ScalarFunction {
                    fun: DateBinGapfill,
                    args,
                } => {
                    if dbg_idx.is_some() {
                        return Err(DataFusionError::Plan(format!(
                            "DATE_BIN_GAPFILL specified more than once"
                        )));
                    }
                    dbg_idx = Some(i);
                    Expr::ScalarFunction {
                        fun: DateBin,
                        args: args.clone(),
                    }
                }
                _ => e.clone(),
            };
            new_aggr_expr.push(new_expr);
            Ok(())
        })?;
    Ok(Some((new_aggr_expr, dbg_idx.unwrap())))
}

fn collect_gap_fns(e: &Expr) -> Result<Vec<BuiltinScalarFunction>> {
    struct Finder {
        fns: Vec<BuiltinScalarFunction>,
    }
    impl ExpressionVisitor for Finder {
        fn pre_visit(mut self, expr: &Expr) -> Result<Recursion<Self>> {
            if let Expr::ScalarFunction {
                fun:
                    f @ (BuiltinScalarFunction::DateBinGapfill | BuiltinScalarFunction::LOCF),
                ..
            } = expr
            {
                self.fns.push(f.clone())
            };
            Ok(Recursion::Continue(self))
        }
    }
    let f = Finder { fns: vec![] };
    let f = e.accept(f)?;
    Ok(f.fns)
}

fn check_node(node: &LogicalPlan) -> Result<()> {
    println!("checking node");
    node.expressions().iter().try_for_each(|expr| {
        let fns = collect_gap_fns(expr)?;
        if !fns.is_empty() {
            Err(DataFusionError::Plan(format!(
                "invalid context for {}",
                fns[0]
            )))
        } else {
            Ok(())
        }
    })?;
    Ok(())
}

/// A logical node that represents the gap filling operation.
///
/// This was made a user-defined node for easier iterative development
/// but it's representation may change. The inner plan node is always
/// an Aggregate node, since it needs similar inputs.
#[derive(Clone, Debug)]
struct GapFill {
    inner: LogicalPlan,
}

impl UserDefinedLogicalNode for GapFill {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        let input = Aggregate::try_from_plan(&self.inner)
            .expect("this should always be an Aggregate node")
            .input
            .as_ref();
        vec![input]
    }

    fn schema(&self) -> &datafusion_common::DFSchemaRef {
        &Aggregate::try_from_plan(&self.inner)
            .expect("this should aways be an Aggregate node")
            .schema
    }

    fn expressions(&self) -> Vec<datafusion_expr::Expr> {
        self.inner.expressions()
    }

    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.inner {
            LogicalPlan::Aggregate(Aggregate {
                ref group_expr,
                ref aggr_expr,
                ..
            }) => write!(f, "GapFill: groupBy=[{group_expr:?}], aggr=[{aggr_expr:?}]"),
            _ => unreachable!(),
        }
    }

    fn from_template(
        &self,
        exprs: &[datafusion_expr::Expr],
        inputs: &[LogicalPlan],
    ) -> std::sync::Arc<dyn UserDefinedLogicalNode> {
        let aggr = Aggregate::try_from_plan(&self.inner)
            .expect("this should aways be an Aggregate node");
        let (group_expr, agg_expr) = exprs.split_at(aggr.group_expr.len());
        let group_expr = Vec::from(group_expr);
        let aggr_expr = Vec::from(agg_expr);
        let new_agg =
            Aggregate::try_new(Arc::new(inputs[0].clone()), group_expr, aggr_expr)
                .expect("should succeed");
        Arc::new(GapFill {
            inner: LogicalPlan::Aggregate(new_agg),
        })
    }
}

#[cfg(test)]
mod test {
    use super::HandleGapFill;

    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use datafusion_common::{Result, ScalarValue};
    use datafusion_expr::{
        avg, call_fn, col, lit, lit_timestamp_nano, logical_plan, Expr, LogicalPlan,
        LogicalPlanBuilder,
    };
    use std::sync::Arc;

    fn table_scan() -> Result<LogicalPlan> {
        let schema = Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new("loc", DataType::Utf8, false),
            Field::new("temp", DataType::Float64, false),
        ]);
        logical_plan::table_scan(Some("temps"), &schema, None)?.build()
    }

    fn date_bin_gapfill(interval: Expr, time: Expr) -> Result<Expr> {
        call_fn(
            "date_bin_gapfill",
            vec![interval, time, lit_timestamp_nano(0)],
        )
    }

    fn assert_optimizer_err(plan: &LogicalPlan, expected: &str) {
        let rule = Arc::new(HandleGapFill {});
        crate::test::assert_optimizer_err(rule, plan, expected)
    }

    fn assert_optimized_plan_eq(plan: &LogicalPlan, expected: &str) -> Result<()> {
        crate::test::assert_optimized_plan_eq(Arc::new(HandleGapFill {}), &plan, expected)
    }

    fn assert_optimization_skipped(plan: &LogicalPlan) -> Result<()> {
        let rule = Arc::new(HandleGapFill {});
        crate::test::assert_optimization_skipped(rule, plan)
    }

    #[test]
    fn misplaced_fns_err() -> Result<()> {
        let scan = table_scan()?;
        let plan = LogicalPlanBuilder::from(scan)
            .filter(
                date_bin_gapfill(
                    lit(ScalarValue::IntervalDayTime(Some(60_0000))),
                    col("temp"),
                )?
                .gt(lit(100.0)),
            )?
            .build()?;
        assert_optimizer_err(
            &plan,
            "Error during planning: invalid context for datebingapfill",
        );
        Ok(())
    }

    #[test]
    fn no_change() -> Result<()> {
        let plan = LogicalPlanBuilder::from(table_scan()?)
            .aggregate(vec![col("loc")], vec![avg(col("temp"))])?
            .build()?;
        assert_optimization_skipped(&plan)?;
        Ok(())
    }

    #[test]
    fn date_bin_gapfill_simple() -> Result<()> {
        let plan = LogicalPlanBuilder::from(table_scan()?)
            .aggregate(
                vec![date_bin_gapfill(
                    lit(ScalarValue::IntervalDayTime(Some(60_000))),
                    col("time"),
                )?],
                vec![avg(col("temp"))],
            )?
            .build()?;

        let expected = "GapFill: groupBy=[[datebin(IntervalDayTime(\"60000\"),temps.time,TimestampNanosecond(0, None))]], aggr=[[AVG(temps.temp)]]\
                      \n  Sort: datebin(IntervalDayTime(\"60000\"),temps.time,TimestampNanosecond(0, None))\
                      \n    Aggregate: groupBy=[[datebin(IntervalDayTime(\"60000\"), temps.time, TimestampNanosecond(0, None))]], aggr=[[AVG(temps.temp)]]\
                      \n      TableScan: temps";
        assert_optimized_plan_eq(&plan, expected)?;
        Ok(())
    }

    #[test]
    fn reordered_sort_exprs() -> Result<()> {
        // grouping by date_bin_gapfill(...), loc
        // but the sort node should have date_bin_gapfill last.
        let plan = LogicalPlanBuilder::from(table_scan()?)
            .aggregate(
                vec![
                    date_bin_gapfill(
                        lit(ScalarValue::IntervalDayTime(Some(60_000))),
                        col("time"),
                    )?,
                    col("loc"),
                ],
                vec![avg(col("temp"))],
            )?
            .build()?;

        let expected = "GapFill: groupBy=[[datebin(IntervalDayTime(\"60000\"),temps.time,TimestampNanosecond(0, None)), temps.loc]], aggr=[[AVG(temps.temp)]]\
                      \n  Sort: temps.loc, datebin(IntervalDayTime(\"60000\"),temps.time,TimestampNanosecond(0, None))\
                      \n    Aggregate: groupBy=[[datebin(IntervalDayTime(\"60000\"), temps.time, TimestampNanosecond(0, None)), temps.loc]], aggr=[[AVG(temps.temp)]]\
                      \n      TableScan: temps";
        assert_optimized_plan_eq(&plan, expected)?;
        Ok(())
    }

    #[test]
    fn double_date_bin_gapfill() -> Result<()> {
        let plan = LogicalPlanBuilder::from(table_scan()?)
            .aggregate(
                vec![
                    date_bin_gapfill(
                        lit(ScalarValue::IntervalDayTime(Some(60_000))),
                        col("time"),
                    )?,
                    date_bin_gapfill(
                        lit(ScalarValue::IntervalDayTime(Some(30_000))),
                        col("time"),
                    )?,
                ],
                vec![avg(col("temp"))],
            )?
            .build()?;
        assert_optimizer_err(
            &plan,
            "Error during planning: DATE_BIN_GAPFILL specified more than once",
        );
        Ok(())
    }
}
