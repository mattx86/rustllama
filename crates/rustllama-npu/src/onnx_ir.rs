//! Minimal ONNX (protobuf) model synthesis for a single `Gemm` op — the AMD
//! analog of the Intel OpenVINO XML IR ([`crate::offload`]'s `matmul_ir`).
//!
//! The Ryzen AI / XDNA NPU is reached through ONNX Runtime + the VitisAI (Ryzen
//! AI) execution provider, which takes a serialized ONNX **model** (protobuf),
//! not a hand-written XML like OpenVINO. We only ever need ONE op (a batched
//! GEMM), so rather than pull in a protobuf code-generator we hand-encode the
//! handful of `onnx.proto` messages that op requires. The schema fields used
//! (from `onnx.proto`):
//!
//! - `ModelProto { ir_version=1:int64, producer_name=2:string, graph=7:msg,
//!    opset_import=8:repeated msg }`
//! - `OperatorSetIdProto { domain=1:string, version=2:int64 }`
//! - `GraphProto { node=1:repeated msg, name=2:string, input=11:repeated msg,
//!    output=12:repeated msg }`
//! - `NodeProto { input=1:repeated string, output=2:repeated string,
//!    name=3:string, op_type=4:string, attribute=5:repeated msg }`
//! - `AttributeProto { name=1:string, f=2:float, i=3:int64, type=20:int64 }`
//! - `ValueInfoProto { name=1:string, type=2:msg }`
//! - `TypeProto { tensor_type=1:msg }`, `Tensor { elem_type=1:int32,
//!    shape=2:msg }`, `TensorShapeProto { dim=1:repeated msg }`, `Dimension { dim_value=1:int64 }`
//!
//! We emit a `Gemm` node (not `MatMul`) because Gemm carries `transB`, which
//! expresses the `out[n,m] = x[n,k]·w[m,k]ᵀ` weight layout the models crate uses
//! without a separate Transpose op. `alpha=1, beta=0`, no `C` input (optional
//! since opset 11). WRITE-BLIND: the bytes are validated by ORT parsing them on
//! Ryzen AI hardware (fail-soft + parity-gated), the same MO as the OpenVINO XML.

// ---- protobuf wire-format primitives (wire types: 0 varint, 2 len-delimited,
// ---- 5 fixed32) ----

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            break;
        }
    }
}

fn put_tag(out: &mut Vec<u8>, field: u32, wire: u32) {
    put_varint(out, ((field as u64) << 3) | wire as u64);
}

/// A length-delimited field (string / bytes / embedded message).
fn put_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    put_tag(out, field, 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn put_string(out: &mut Vec<u8>, field: u32, s: &str) {
    put_bytes(out, field, s.as_bytes());
}

fn put_msg(out: &mut Vec<u8>, field: u32, msg: &[u8]) {
    put_bytes(out, field, msg);
}

fn put_int(out: &mut Vec<u8>, field: u32, v: i64) {
    put_tag(out, field, 0);
    put_varint(out, v as u64);
}

fn put_float(out: &mut Vec<u8>, field: u32, v: f32) {
    put_tag(out, field, 5);
    out.extend_from_slice(&v.to_le_bytes());
}

// ---- ONNX message builders ----

/// `ValueInfoProto` for a FLOAT tensor named `name` with static `dims`.
fn value_info(name: &str, dims: &[usize]) -> Vec<u8> {
    // TensorShapeProto { repeated Dimension dim = 1 { dim_value = 1 } }
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        put_int(&mut dim, 1, d as i64); // dim_value
        put_msg(&mut shape, 1, &dim); // dim
    }
    // Tensor { elem_type = 1 (FLOAT=1); shape = 2 }
    let mut tensor = Vec::new();
    put_int(&mut tensor, 1, 1); // elem_type FLOAT
    put_msg(&mut tensor, 2, &shape);
    // TypeProto { tensor_type = 1 }
    let mut typ = Vec::new();
    put_msg(&mut typ, 1, &tensor);
    // ValueInfoProto { name = 1; type = 2 }
    let mut vi = Vec::new();
    put_string(&mut vi, 1, name);
    put_msg(&mut vi, 2, &typ);
    vi
}

/// An INT `AttributeProto` (`type` = INT = 2).
fn attr_int(name: &str, i: i64) -> Vec<u8> {
    let mut a = Vec::new();
    put_string(&mut a, 1, name);
    put_int(&mut a, 3, i); // i
    put_int(&mut a, 20, 2); // type = INT
    a
}

/// A FLOAT `AttributeProto` (`type` = FLOAT = 1).
fn attr_float(name: &str, f: f32) -> Vec<u8> {
    let mut a = Vec::new();
    put_string(&mut a, 1, name);
    put_float(&mut a, 2, f); // f
    put_int(&mut a, 20, 1); // type = FLOAT
    a
}

/// Serialize a single-`Gemm` ONNX model computing, for inputs `A[ar,ac]` and
/// `B[br,bc]`, the output `Y = A · (transpose_b ? Bᵀ : B)` (alpha=1, beta=0, no
/// C). Output shape is `ar × (transpose_b ? br : bc)`. Static shapes (the NPU
/// compiler wants them).
pub fn gemm_model(ar: usize, ac: usize, br: usize, bc: usize, transpose_b: bool) -> Vec<u8> {
    let (out_r, out_c) = if transpose_b { (ar, br) } else { (ar, bc) };

    // NodeProto { input*=1, output*=2, name=3, op_type=4, attribute*=5 }
    let mut node = Vec::new();
    put_string(&mut node, 1, "A"); // input A
    put_string(&mut node, 1, "B"); // input B
    put_string(&mut node, 2, "Y"); // output
    put_string(&mut node, 3, "rsl_gemm");
    put_string(&mut node, 4, "Gemm");
    put_msg(&mut node, 5, &attr_int("transA", 0));
    put_msg(
        &mut node,
        5,
        &attr_int("transB", if transpose_b { 1 } else { 0 }),
    );
    put_msg(&mut node, 5, &attr_float("alpha", 1.0));
    put_msg(&mut node, 5, &attr_float("beta", 0.0));

    // GraphProto { node*=1, name=2, input*=11, output*=12 }
    let mut graph = Vec::new();
    put_msg(&mut graph, 1, &node);
    put_string(&mut graph, 2, "rsl_graph");
    put_msg(&mut graph, 11, &value_info("A", &[ar, ac]));
    put_msg(&mut graph, 11, &value_info("B", &[br, bc]));
    put_msg(&mut graph, 12, &value_info("Y", &[out_r, out_c]));

    // OperatorSetIdProto { domain=1, version=2 } — default domain, opset 13.
    let mut opset = Vec::new();
    put_string(&mut opset, 1, "");
    put_int(&mut opset, 2, 13);

    // ModelProto { ir_version=1, producer_name=2, graph=7, opset_import*=8 }
    let mut model = Vec::new();
    put_int(&mut model, 1, 7); // ONNX IR version 7 (opset 13-compatible)
    put_string(&mut model, 2, "rustllama");
    put_msg(&mut model, 7, &graph);
    put_msg(&mut model, 8, &opset);
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal protobuf reader, just enough to structurally validate what
    // `gemm_model` emits (so a wire-format mistake fails HERE, not silently on
    // hardware). Returns (field_number, wire_payload) pairs at one level.
    fn read_varint(buf: &[u8], pos: &mut usize) -> u64 {
        let mut shift = 0;
        let mut v = 0u64;
        loop {
            let b = buf[*pos];
            *pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        v
    }

    enum Val {
        Varint(u64),
        Bytes(Vec<u8>),
        Fixed32,
    }

    fn parse(buf: &[u8]) -> Vec<(u32, Val)> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < buf.len() {
            let tag = read_varint(buf, &mut pos);
            let field = (tag >> 3) as u32;
            let wire = (tag & 7) as u32;
            match wire {
                0 => out.push((field, Val::Varint(read_varint(buf, &mut pos)))),
                2 => {
                    let len = read_varint(buf, &mut pos) as usize;
                    let b = buf[pos..pos + len].to_vec();
                    pos += len;
                    out.push((field, Val::Bytes(b)));
                }
                5 => {
                    pos += 4;
                    out.push((field, Val::Fixed32));
                }
                _ => panic!("unexpected wire type {wire}"),
            }
        }
        out
    }

    fn get_bytes(fields: &[(u32, Val)], f: u32) -> &[u8] {
        fields
            .iter()
            .find_map(|(n, v)| match v {
                Val::Bytes(b) if *n == f => Some(b.as_slice()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no bytes field {f}"))
    }

    fn get_int(fields: &[(u32, Val)], f: u32) -> u64 {
        fields
            .iter()
            .find_map(|(n, v)| match v {
                Val::Varint(x) if *n == f => Some(*x),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no varint field {f}"))
    }

    fn get_strings(fields: &[(u32, Val)], f: u32) -> Vec<String> {
        fields
            .iter()
            .filter_map(|(n, v)| match v {
                Val::Bytes(b) if *n == f => Some(String::from_utf8_lossy(b).into_owned()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn varint_roundtrips() {
        for v in [0u64, 1, 127, 128, 300, 16384, 1 << 31, u32::MAX as u64] {
            let mut buf = Vec::new();
            put_varint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_varint(&buf, &mut pos), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn gemm_model_decodes_to_expected_graph() {
        // out[4,8] = A[4,16] · B[8,16]ᵀ (the xwt layout: transpose_b=true).
        let bytes = gemm_model(4, 16, 8, 16, true);
        let model = parse(&bytes);

        assert_eq!(get_int(&model, 1), 7, "ir_version");
        // graph (field 7)
        let graph = parse(get_bytes(&model, 7));
        // node (field 1)
        let node = parse(get_bytes(&graph, 1));
        assert_eq!(get_strings(&node, 1), vec!["A", "B"], "node inputs");
        assert_eq!(get_strings(&node, 2), vec!["Y"], "node output");
        assert_eq!(get_strings(&node, 4), vec!["Gemm"], "op_type");

        // Exactly one transB attribute with value 1.
        let attrs: Vec<Vec<(u32, Val)>> = node
            .iter()
            .filter_map(|(n, v)| match v {
                Val::Bytes(b) if *n == 5 => Some(parse(b)),
                _ => None,
            })
            .collect();
        let transb = attrs
            .iter()
            .find(|a| get_strings(a, 1) == vec!["transB"])
            .expect("transB attr");
        assert_eq!(get_int(transb, 3), 1, "transB=1");

        // Graph inputs A[4,16], B[8,16]; output Y[4,8].
        let inputs = get_strings(&graph, 11)
            .iter()
            .map(|_| ())
            .collect::<Vec<_>>();
        assert_eq!(inputs.len(), 2, "two graph inputs");
        // Output ValueInfo (field 12) exists.
        assert!(graph.iter().any(|(n, _)| *n == 12), "has graph output");
    }

    #[test]
    fn gemm_model_plain_vs_transposed_differ() {
        // transpose_b flips the output columns (bc vs br) → different bytes.
        assert_ne!(
            gemm_model(4, 16, 16, 8, false),
            gemm_model(4, 16, 8, 16, true)
        );
    }
}
