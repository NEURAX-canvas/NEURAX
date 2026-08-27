//! Real per-family shape inference for `TensorPass`.
//!
//! Before this module, any layer whose client-declared shape was empty got
//! the same placeholder — `[batch, seq, hidden]` — regardless of its actual
//! type: a Conv2D layer was treated identically to an attention layer. This
//! is a port of the shape engine already built and verified this session in
//! `neurax-ui/src/utils/neuraxCompiler.ts`, now the canonical implementation
//! instead of a client-side guess. See the Phase 1 plan for scope: CNN (and
//! GAN/diffusion, which share the same `[B,C,H,W]` convention) and the
//! sequence families (transformer/moe/ssm/rnn). GNN and everything else
//! pass their input shape through unchanged, same as before this module.

use std::collections::HashMap;

use neurax_formulas::conv2d_output_shape;
use neurax_parser::{LayerParams, LayerType, ModelConfig, ModelType};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use crate::graph::GraphIR;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShapeFamily {
    /// `[batch, channels, height, width]` — cnn/gan/diffusion.
    Image,
    /// `[batch, sequence, hidden]` — transformer/moe/ssm/rnn.
    Sequence,
    /// Not yet covered (gnn and the rest) — input shape passes through
    /// unchanged, exactly like before this module existed.
    Passthrough,
}

fn shape_family(model_type: ModelType) -> ShapeFamily {
    match model_type {
        ModelType::Cnn | ModelType::Gan | ModelType::Diffusion => ShapeFamily::Image,
        ModelType::Transformer | ModelType::Moe | ModelType::Ssm | ModelType::Rnn => {
            ShapeFamily::Sequence
        }
        _ => ShapeFamily::Passthrough,
    }
}

/// The shape a root node (no real predecessor) starts from, when the client
/// sent none — read from the same `data.image_*` / `global_params` fields
/// `operator/pass.rs`'s FLOPs dispatch already reads for the same purpose.
fn entry_shape(family: ShapeFamily, batch: usize, config: &ModelConfig) -> Vec<usize> {
    match family {
        ShapeFamily::Image => {
            let d = &config.data;
            vec![
                batch,
                d.image_channels.unwrap_or(3),
                d.image_height.unwrap_or(224),
                d.image_width.unwrap_or(224),
            ]
        }
        ShapeFamily::Sequence => {
            let gp = &config.model.global_params;
            vec![
                batch,
                gp.sequence_length.unwrap_or(512),
                gp.embedding_dim.unwrap_or(512),
            ]
        }
        ShapeFamily::Passthrough => vec![],
    }
}

/// `kernel_h`/`kernel_w` when given, else the shared `kernel_size` (a square
/// kernel), else `default` — the same precedence `operator/pass.rs`'s own
/// Conv FLOPs dispatch uses.
fn square_kernel(params: &LayerParams, default: usize) -> (usize, usize) {
    let kh = params.kernel_h.or(params.kernel_size).unwrap_or(default);
    let kw = params.kernel_w.or(params.kernel_size).unwrap_or(kh);
    (kh, kw)
}

fn image_output_shape(layer_type: LayerType, params: &LayerParams, input: &[usize]) -> Vec<usize> {
    // Dense is the one type meant to consume either a still-spatial (4D) or
    // an already-flat (2D, from an earlier Dense) input and always produce
    // 2D — it must not be gated behind the "input must be 4D" shape below,
    // which is only meaningful for the genuinely spatial ops.
    if let LayerType::Dense = layer_type {
        let (batch, flat) = match input.len() {
            4 => (input[0], input[1] * input[2] * input[3]),
            2 => (input[0], input[1]),
            _ => return input.to_vec(),
        };
        let out = params.out_features.unwrap_or(flat);
        return vec![batch, out];
    }

    if input.len() != 4 {
        return input.to_vec();
    }
    let (batch, in_ch, h, w) = (input[0], input[1], input[2], input[3]);
    match layer_type {
        LayerType::Conv => {
            let (kh, kw) = square_kernel(params, 3);
            let stride = params.stride.unwrap_or(1);
            let padding = params.padding.unwrap_or(0);
            let (out_h, out_w) = conv2d_output_shape(h, w, kh, kw, stride, padding);
            let out_ch = params.out_channels.unwrap_or(in_ch);
            vec![batch, out_ch, out_h.max(1), out_w.max(1)]
        }
        LayerType::Pooling => {
            if params.pool_type.as_deref() == Some("global") {
                return vec![batch, in_ch, 1, 1];
            }
            let (kh, kw) = square_kernel(params, 2);
            let stride = params.stride.unwrap_or(2);
            let padding = params.padding.unwrap_or(0);
            let (out_h, out_w) = conv2d_output_shape(h, w, kh, kw, stride, padding);
            vec![batch, in_ch, out_h.max(1), out_w.max(1)]
        }
        _ => input.to_vec(),
    }
}

fn sequence_output_shape(
    layer_type: LayerType,
    params: &LayerParams,
    input: &[usize],
) -> Vec<usize> {
    if layer_type == LayerType::Embedding && input.len() == 2 {
        let hidden = params.embedding_dim.or(params.hidden_size).unwrap_or(512);
        return vec![input[0], input[1], hidden];
    }
    if input.len() != 3 {
        return input.to_vec();
    }
    let (batch, seq) = (input[0], input[1]);
    // A node's own stated width wins — the same hidden-dim-threading logic
    // already shipped in `neurax-agent/budget_check.py::spec_to_topology`
    // (added earlier this session), now the canonical implementation: a
    // node that never states one just carries forward what it received.
    let hidden = params
        .hidden_size
        .or(params.embedding_dim)
        .or(params.out_features)
        .unwrap_or(input[2]);
    vec![batch, seq, hidden]
}

fn infer_output_shape(
    layer_type: LayerType,
    family: ShapeFamily,
    params: &LayerParams,
    input: &[usize],
) -> Vec<usize> {
    match family {
        ShapeFamily::Image => image_output_shape(layer_type, params, input),
        ShapeFamily::Sequence => sequence_output_shape(layer_type, params, input),
        ShapeFamily::Passthrough => input.to_vec(),
    }
}

/// Walk the graph in topological order, resolving each node's (input,
/// output) shape from its real predecessor's actual output — not array
/// position. A client-declared shape is never overwritten, only filled in
/// when empty, so a caller that already sends its own shapes sees no change.
pub fn infer_shapes(
    graph: &GraphIR,
    config: &ModelConfig,
) -> HashMap<String, (Vec<usize>, Vec<usize>)> {
    let family = shape_family(config.model.model_type);
    let batch = config.training.batch_size;
    let params_by_id: HashMap<&str, &LayerParams> = config
        .model
        .layers
        .iter()
        .map(|l| (l.id.as_str(), &l.params))
        .collect();

    let mut resolved_output: HashMap<NodeIndex, Vec<usize>> = HashMap::new();
    let mut out: HashMap<String, (Vec<usize>, Vec<usize>)> = HashMap::new();

    for &idx in &graph.topo_order {
        let Some(node) = graph.dag.node_weight(idx) else {
            continue;
        };

        let declared_input = node.input_shapes.first().cloned().unwrap_or_default();
        let input_shape = if !declared_input.is_empty() {
            declared_input
        } else {
            let mut incoming = graph.dag.edges_directed(idx, Direction::Incoming);
            match incoming.next() {
                Some(edge) => resolved_output
                    .get(&edge.source())
                    .cloned()
                    .unwrap_or_default(),
                None => entry_shape(family, batch, config),
            }
        };

        let output_shape = if !node.output_shape.is_empty() {
            node.output_shape.clone()
        } else {
            let default_params = LayerParams::default();
            let params = params_by_id
                .get(node.layer_id.as_str())
                .copied()
                .unwrap_or(&default_params);
            infer_output_shape(node.layer_type, family, params, &input_shape)
        };

        resolved_output.insert(idx, output_shape.clone());
        out.insert(node.layer_id.clone(), (input_shape, output_shape));
    }

    out
}
