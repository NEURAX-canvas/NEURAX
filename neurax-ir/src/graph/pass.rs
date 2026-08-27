//! Graph IR pass

use super::{GraphEdge, GraphIR, GraphMetrics, GraphNode};
use crate::architecture::ArchitectureIR;
use crate::error::GraphError;
use crate::traits::IrPass;
use crate::NeuraxContext;
use neurax_formulas::dtype_bytes;
use petgraph::graph::NodeIndex;

/// Graph pass implementation
pub struct GraphPass;

impl IrPass for GraphPass {
    type Input = ArchitectureIR;
    type Output = GraphIR;
    type Metrics = GraphMetrics;
    type PassError = GraphError;

    fn name(&self) -> &'static str {
        "GraphIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        if input.layers.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        let mut graph = GraphIR::new();

        // Add all layers as nodes first — edges (below) need every node's
        // index to already exist, whichever source decides them.
        for layer in &input.layers {
            let node = GraphNode {
                layer_id: layer.id.clone(),
                layer_type: layer.layer_type,
                flops_approx: 0.0, // Will be computed in Operator IR
                input_shapes: if !layer.input_shape.is_empty() {
                    vec![layer.input_shape.clone()]
                } else {
                    vec![]
                },
                output_shape: layer.output_shape.clone(),
                param_count: layer.param_count,
            };
            graph.add_node(node);
        }

        // A client's real graph wins when it sends one. Absent that (every
        // fixture shipped before this field existed, and any client that
        // still doesn't send one), fall back to the original assumption:
        // each layer's only predecessor is the one before it in the array.
        if !ctx.config.model.connections.is_empty() {
            for conn in &ctx.config.model.connections {
                let (Some(&from), Some(&to)) = (
                    graph.node_index.get(&conn.from),
                    graph.node_index.get(&conn.to),
                ) else {
                    continue; // an edge naming an id absent from `layers` — ignore, don't fail the whole graph
                };
                let to_shape = graph
                    .dag
                    .node_weight(to)
                    .map(|n| n.input_shapes.first().cloned().unwrap_or_default())
                    .unwrap_or_default();
                let edge = GraphEdge {
                    tensor_shape: to_shape.clone(),
                    dtype: input.training_config.precision.clone(),
                    size_bytes: calculate_tensor_size(&to_shape, &input.training_config.precision),
                };
                graph.add_edge(from, to, edge);
            }
        } else {
            let mut prev_idx: Option<NodeIndex> = None;
            for layer in &input.layers {
                let idx = graph.node_index[&layer.id];
                if let Some(prev) = prev_idx {
                    let edge = GraphEdge {
                        tensor_shape: layer.input_shape.clone(),
                        dtype: input.training_config.precision.clone(),
                        size_bytes: calculate_tensor_size(
                            &layer.input_shape,
                            &input.training_config.precision,
                        ),
                    };
                    graph.add_edge(prev, idx, edge);
                }
                prev_idx = Some(idx);
            }
        }

        // Compute topological order
        graph
            .compute_topo_order()
            .map_err(GraphError::TopologicalSortFailed)?;

        Ok(graph)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        _ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let metrics = GraphMetrics {
            graph_depth: output.calculate_depth(),
            total_operations: output.dag.node_count(),
            total_intermediate_tensors: output.dag.edge_count(),
            edge_count: output.dag.edge_count(),
            parallel_paths: vec![], // Computed later for non-sequential models
            critical_path_length: output.topo_order.len(),
        };

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(
        &self,
        output: &Self::Output,
        metrics: &Self::Metrics,
    ) -> Result<(), Self::PassError> {
        if output.has_cycle() {
            return Err(GraphError::CycleDetected);
        }
        if metrics.total_operations == 0 {
            return Err(GraphError::EmptyGraph);
        }
        Ok(())
    }
}

fn calculate_tensor_size(shape: &[usize], dtype: &str) -> u64 {
    if shape.is_empty() {
        return 0;
    }
    let elements: usize = shape.iter().product();
    (elements * dtype_bytes(dtype)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use neurax_parser::parse_model_config;

    fn create_test_config() -> neurax_parser::ModelConfig {
        let json = r#"{
            "schema_version": "1.0",
            "model": {
                "name": "test",
                "type": "transformer",
                "layers": [{"id": "test_layer", "layer_type": "dense", "input_shape": [32, 512], "output_shape": [32, 256]}]
            },
            "training": {"batch_size": 32, "precision": "fp16"},
            "hardware": {"gpus": [{"name": "A100", "count": 1}]}
        }"#;
        parse_model_config(json).unwrap()
    }

    #[test]
    fn test_graph_construction() {
        let mut arch_ir = ArchitectureIR::default();
        arch_ir.layers.push(crate::architecture::LayerDef {
            id: "layer1".to_string(),
            layer_type: neurax_parser::LayerType::Dense,
            input_shape: vec![32, 512],
            output_shape: vec![32, 256],
            params: neurax_parser::LayerParams::default(),
            custom_equations: None,
            param_count: 131072,
        });
        arch_ir.layers.push(crate::architecture::LayerDef {
            id: "layer2".to_string(),
            layer_type: neurax_parser::LayerType::Dense,
            input_shape: vec![32, 256],
            output_shape: vec![32, 128],
            params: neurax_parser::LayerParams::default(),
            custom_equations: None,
            param_count: 32896,
        });

        let ctx = NeuraxContext::new(create_test_config());
        let pass = GraphPass;
        let mut graph = pass.build(&arch_ir, &ctx).unwrap();
        let metrics = pass.compute_metrics(&mut graph, &ctx).unwrap();

        assert_eq!(metrics.total_operations, 2);
        assert!(!graph.has_cycle());
    }
}
