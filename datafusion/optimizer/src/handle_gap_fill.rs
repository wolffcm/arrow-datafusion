use datafusion_expr::{Aggregate, LogicalPlan, UserDefinedLogicalNode};
use std::sync::Arc;

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
