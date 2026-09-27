//! 4-bit dequantization kernels for AWQ and GPTQ safetensors weights.
//!
//! Both formats pack 8 4-bit weights into one `int32` and apply
//! per-group scale + zero-point dequantization, but they differ on:
//!
//! * **Packing axis.** AWQ packs along the *output* dimension
//!   (`qweight.shape = [in_features, out_features / 8]`); GPTQ packs
//!   along the *input* dimension (`qweight.shape = [in_features / 8,
//!   out_features]`). The bit order inside the int32 is little-endian
//!   nibbles in both cases: bits `0..4` hold lane 0, bits `4..8` hold
//!   lane 1, …
//! * **Group mapping.** AWQ uses fixed-stride groups
//!   (`g = i / group_size`); GPTQ uses an explicit per-row remap
//!   table `g_idx[i]` so the algorithm can apply "actorder" grouping
//!   from the calibration step.
//! * **Zero-point offset.** GPTQ stores `qzero = actual_zero - 1`
//!   (so that an `int4` zero of 0 represents an actual offset of 1);
//!   the dequant formula adds 1 before subtracting. AWQ uses the
//!   stored value verbatim.
//!
//! Output layout: row-major `[in_features, out_features]` `f16`,
//! matching the unpacked PyTorch convention (`weight.T` of the
//! corresponding nn.Linear). The A-2 conversion path will transpose
//! into rustllama's `[out_features, in_features]` weight convention.

use half::f16;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DequantError {
    #[error(
        "qweight length {got} doesn't match expected {expected} for shape \
         [{in_features}, {out_features}/{lanes_per_pack}]"
    )]
    QweightShape {
        got: usize,
        expected: usize,
        in_features: usize,
        out_features: usize,
        lanes_per_pack: usize,
    },
    #[error(
        "scales length {got} doesn't match expected {expected} for shape \
         [n_groups={n_groups}, out_features={out_features}]"
    )]
    ScalesShape {
        got: usize,
        expected: usize,
        n_groups: usize,
        out_features: usize,
    },
    #[error(
        "qzeros length {got} doesn't match expected {expected} for shape \
         [n_groups={n_groups}, out_features={out_features}/{lanes_per_pack}]"
    )]
    QzerosShape {
        got: usize,
        expected: usize,
        n_groups: usize,
        out_features: usize,
        lanes_per_pack: usize,
    },
    #[error(
        "g_idx length {got} doesn't match expected in_features={in_features}"
    )]
    GIdxShape { got: usize, in_features: usize },
    #[error(
        "g_idx[{idx}] = {value} but only {n_groups} groups exist; \
         corrupt GPTQ file"
    )]
    GIdxOutOfRange { idx: usize, value: i32, n_groups: usize },
    #[error("out_features={out_features} is not a multiple of 8 (4-bit lanes per int32)")]
    OutFeaturesNotPacked { out_features: usize },
    #[error("in_features={in_features} is not a multiple of 8 (GPTQ packs along in_features)")]
    InFeaturesNotPacked { in_features: usize },
    #[error(
        "group_size {group_size} does not divide in_features {in_features}"
    )]
    GroupSizeMisaligned { group_size: usize, in_features: usize },
}

/// AWQ INT4 → FP16 dequantization.
///
/// # Layout (AWQ convention, matching autoawq's `WQLinear_GEMM`)
///
/// - `qweight`: `[in_features, out_features / 8]` row-major i32. Each
///   int32 packs 8 4-bit weights along the **output** axis:
///   `qweight[i, j]` carries `w[i, j*8 + 0..8]` in nibbles
///   `0..4, 4..8, 8..12, …, 28..32`.
/// - `scales`: `[in_features / group_size, out_features]` row-major
///   `f16`.
/// - `qzeros`: `[in_features / group_size, out_features / 8]` row-major
///   i32. Same packing as `qweight` but along the output axis only
///   (one group per row).
/// - `group_size` is the group height in the **input** dimension.
///   Common values: 32, 64, 128 (Qwen2.5-Coder AWQ uses 128).
///
/// # Formula
///
/// For each `(i, j)`:
///
/// ```text
///   group       = i / group_size
///   weight_int4 = nibble j%8 of qweight[i, j/8]
///   zero_int4   = nibble j%8 of qzeros[group, j/8]
///   out[i, j]   = (weight_int4 - zero_int4) * scales[group, j]
/// ```
///
/// AWQ does NOT apply the GPTQ `+1` to the zero point.
///
/// Output is `[in_features, out_features]` row-major `f16`.
pub fn dequant_awq_int4_to_f16(
    qweight: &[i32],
    scales: &[f16],
    qzeros: &[i32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> Result<Vec<f16>, DequantError> {
    const LANES: usize = 8;
    if out_features % LANES != 0 {
        return Err(DequantError::OutFeaturesNotPacked { out_features });
    }
    if in_features % group_size != 0 {
        return Err(DequantError::GroupSizeMisaligned {
            group_size,
            in_features,
        });
    }
    let n_groups = in_features / group_size;
    let out_packs = out_features / LANES;
    let expected_qw = in_features * out_packs;
    if qweight.len() != expected_qw {
        return Err(DequantError::QweightShape {
            got: qweight.len(),
            expected: expected_qw,
            in_features,
            out_features,
            lanes_per_pack: LANES,
        });
    }
    let expected_sc = n_groups * out_features;
    if scales.len() != expected_sc {
        return Err(DequantError::ScalesShape {
            got: scales.len(),
            expected: expected_sc,
            n_groups,
            out_features,
        });
    }
    let expected_qz = n_groups * out_packs;
    if qzeros.len() != expected_qz {
        return Err(DequantError::QzerosShape {
            got: qzeros.len(),
            expected: expected_qz,
            n_groups,
            out_features,
            lanes_per_pack: LANES,
        });
    }

    let mut out = vec![f16::ZERO; in_features * out_features];
    for i in 0..in_features {
        let g = i / group_size;
        for jp in 0..out_packs {
            let w_pack = qweight[i * out_packs + jp] as u32;
            let z_pack = qzeros[g * out_packs + jp] as u32;
            for k in 0..LANES {
                let shift = (k * 4) as u32;
                let w_int4 = ((w_pack >> shift) & 0xF) as i32;
                let z_int4 = ((z_pack >> shift) & 0xF) as i32;
                let j = jp * LANES + k;
                let scale = scales[g * out_features + j].to_f32();
                let val = (w_int4 - z_int4) as f32 * scale;
                out[i * out_features + j] = f16::from_f32(val);
            }
        }
    }
    Ok(out)
}

/// GPTQ INT4 → FP16 dequantization.
///
/// # Layout (GPTQ convention, matching auto-gptq's `QuantLinear`)
///
/// - `qweight`: `[in_features / 8, out_features]` row-major i32. Each
///   int32 packs 8 4-bit weights along the **input** axis:
///   `qweight[ip, j]` carries `w[ip*8 + 0..8, j]` in nibbles
///   `0..4, 4..8, …`. (Opposite packing axis from AWQ.)
/// - `scales`: `[n_groups, out_features]` row-major `f16`.
/// - `qzeros`: `[n_groups, out_features / 8]` row-major i32. Packed
///   along the output axis, lane order matches AWQ qzeros.
/// - `g_idx`: `[in_features]` int32. `g_idx[i]` is the group index
///   for row `i`; values must lie in `[0, n_groups)`. With "actorder
///   off" it's a plain `i / group_size`; with actorder on it carries
///   the calibration-derived permutation that GPTQ produced.
///
/// # Formula
///
/// For each `(i, j)`:
///
/// ```text
///   group       = g_idx[i]
///   weight_int4 = nibble i%8 of qweight[i/8, j]
///   zero_int4   = nibble j%8 of qzeros[group, j/8]
///   out[i, j]   = (weight_int4 - (zero_int4 + 1)) * scales[group, j]
/// ```
///
/// The `+1` on `zero_int4` is GPTQ's stored-as-`actual-1` convention.
///
/// Output is `[in_features, out_features]` row-major `f16`.
pub fn dequant_gptq_int4_to_f16(
    qweight: &[i32],
    scales: &[f16],
    qzeros: &[i32],
    g_idx: &[i32],
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f16>, DequantError> {
    const LANES: usize = 8;
    if in_features % LANES != 0 {
        return Err(DequantError::InFeaturesNotPacked { in_features });
    }
    if out_features % LANES != 0 {
        return Err(DequantError::OutFeaturesNotPacked { out_features });
    }
    if g_idx.len() != in_features {
        return Err(DequantError::GIdxShape {
            got: g_idx.len(),
            in_features,
        });
    }
    let in_packs = in_features / LANES;
    let expected_qw = in_packs * out_features;
    if qweight.len() != expected_qw {
        return Err(DequantError::QweightShape {
            got: qweight.len(),
            expected: expected_qw,
            in_features,
            out_features,
            lanes_per_pack: LANES,
        });
    }
    // n_groups derives from scales' total length, not from a separate
    // input — GPTQ models with actorder grouping can have an
    // irregular group count that doesn't simply equal
    // in_features/group_size.
    if out_features == 0 {
        return Err(DequantError::OutFeaturesNotPacked { out_features });
    }
    if scales.len() % out_features != 0 {
        return Err(DequantError::ScalesShape {
            got: scales.len(),
            expected: 0,
            n_groups: 0,
            out_features,
        });
    }
    let n_groups = scales.len() / out_features;
    let out_packs = out_features / LANES;
    let expected_qz = n_groups * out_packs;
    if qzeros.len() != expected_qz {
        return Err(DequantError::QzerosShape {
            got: qzeros.len(),
            expected: expected_qz,
            n_groups,
            out_features,
            lanes_per_pack: LANES,
        });
    }
    for (idx, &g) in g_idx.iter().enumerate() {
        if g < 0 || (g as usize) >= n_groups {
            return Err(DequantError::GIdxOutOfRange {
                idx,
                value: g,
                n_groups,
            });
        }
    }

    let mut out = vec![f16::ZERO; in_features * out_features];
    for i in 0..in_features {
        let g = g_idx[i] as usize;
        let ip = i / LANES;
        let in_lane = (i % LANES) as u32;
        for j in 0..out_features {
            let w_pack = qweight[ip * out_features + j] as u32;
            let w_int4 = ((w_pack >> (in_lane * 4)) & 0xF) as i32;
            let jp = j / LANES;
            let out_lane = (j % LANES) as u32;
            let z_pack = qzeros[g * out_packs + jp] as u32;
            let z_int4 = ((z_pack >> (out_lane * 4)) & 0xF) as i32;
            let scale = scales[g * out_features + j].to_f32();
            let val = (w_int4 - (z_int4 + 1)) as f32 * scale;
            out[i * out_features + j] = f16::from_f32(val);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack 8 4-bit values (in lane order 0..8) into one int32 using
    /// the little-endian nibble layout AWQ and GPTQ both use.
    fn pack_int4_lane(lanes: [u8; 8]) -> i32 {
        let mut acc: u32 = 0;
        for (k, v) in lanes.iter().enumerate() {
            let nibble = (*v as u32) & 0xF;
            acc |= nibble << (k * 4);
        }
        acc as i32
    }

    /// Reference AWQ dequant of one row segment: returns the
    /// expected f16 row for `(i, j_base..j_base+8)` given the int4
    /// weight nibbles, int4 zero nibbles, and the per-output-column
    /// scale row.
    fn awq_reference_row_chunk(
        w_int4: [u8; 8],
        z_int4: [u8; 8],
        scale_row: [f32; 8],
    ) -> [f16; 8] {
        let mut out = [f16::ZERO; 8];
        for k in 0..8 {
            let val = (w_int4[k] as i32 - z_int4[k] as i32) as f32 * scale_row[k];
            out[k] = f16::from_f32(val);
        }
        out
    }

    #[test]
    fn awq_dequant_single_int32_block_matches_formula() {
        // Smallest possible AWQ tensor: in_features = group_size = 1,
        // out_features = 8 (one pack). Pin the per-nibble formula.
        let in_f = 1;
        let out_f = 8;
        let group_size = 1;
        // Weights 0..8 in lane order: nibbles [0,1,2,3,4,5,6,7].
        let qw_packed = pack_int4_lane([0, 1, 2, 3, 4, 5, 6, 7]);
        // Zeros: constant 3 across all 8 lanes.
        let qz_packed = pack_int4_lane([3; 8]);
        // Scales: 0.5 for each output column.
        let scales = vec![f16::from_f32(0.5); 8];
        let got = dequant_awq_int4_to_f16(
            &[qw_packed],
            &scales,
            &[qz_packed],
            in_f,
            out_f,
            group_size,
        )
        .unwrap();
        let exp = awq_reference_row_chunk(
            [0, 1, 2, 3, 4, 5, 6, 7],
            [3; 8],
            [0.5; 8],
        );
        for k in 0..8 {
            assert_eq!(
                got[k].to_bits(),
                exp[k].to_bits(),
                "lane {k}: got {} vs exp {}",
                got[k].to_f32(),
                exp[k].to_f32(),
            );
        }
    }

    #[test]
    fn awq_dequant_handles_multiple_groups() {
        // in_features = 4, group_size = 2 → 2 groups. out_features = 8
        // (1 pack). Different scale + zero per group; verify the
        // i / group_size mapping picks the right row.
        let in_f = 4;
        let out_f = 8;
        let group_size = 2;
        // qweight: 4 rows × 1 pack. Row r has lanes (r+k) for k in 0..8.
        let qweight: Vec<i32> = (0..4u8)
            .map(|r| pack_int4_lane([r, r + 1, r + 2, r + 3, r + 4, r + 5, r + 6, r + 7]))
            .collect();
        // qzeros: 2 groups, 1 pack each. Group g zeros are all g+1.
        let qzeros = vec![
            pack_int4_lane([1; 8]),
            pack_int4_lane([2; 8]),
        ];
        // scales: 2 groups × 8 outs. Group g has scale g+1 across all outs.
        let scales: Vec<f16> = (0..2u32)
            .flat_map(|g| std::iter::repeat(f16::from_f32((g + 1) as f32)).take(8))
            .collect();

        let got =
            dequant_awq_int4_to_f16(&qweight, &scales, &qzeros, in_f, out_f, group_size)
                .unwrap();
        // Row 0: group 0, w=[0,1,2,3,4,5,6,7], z=1, s=1 -> [-1,0,1,2,3,4,5,6]
        // Row 1: group 0, w=[1,2,3,4,5,6,7,8], z=1, s=1 -> [0,1,2,3,4,5,6,7]
        // Row 2: group 1, w=[2,3,4,5,6,7,8,9], z=2, s=2 -> [0,2,4,6,8,10,12,14]
        // Row 3: group 1, w=[3,4,5,6,7,8,9,10], z=2, s=2 -> [2,4,6,8,10,12,14,16]
        let expected = [
            [-1.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        ];
        for r in 0..4 {
            for c in 0..8 {
                let g = got[r * 8 + c].to_f32();
                let e = expected[r][c];
                assert!(
                    (g - e).abs() < 1e-3,
                    "row {r}, col {c}: got {g}, expected {e}",
                );
            }
        }
    }

    #[test]
    fn awq_dequant_rejects_shape_mismatch() {
        // qweight has 1 entry but in_features=2 ⇒ expected 2.
        let err = dequant_awq_int4_to_f16(
            &[0i32],
            &[f16::ZERO; 8],
            &[0i32; 1],
            2, // in_features
            8, // out_features
            2, // group_size
        )
        .unwrap_err();
        assert!(matches!(err, DequantError::QweightShape { got: 1, .. }));
    }

    #[test]
    fn awq_dequant_rejects_unpacked_out_features() {
        let err = dequant_awq_int4_to_f16(
            &[0i32; 0],
            &[f16::ZERO; 5],
            &[0i32; 0],
            1, // in_features
            5, // out_features — not a multiple of 8
            1,
        )
        .unwrap_err();
        assert!(matches!(err, DequantError::OutFeaturesNotPacked { .. }));
    }

    #[test]
    fn awq_dequant_rejects_misaligned_group_size() {
        let err = dequant_awq_int4_to_f16(
            &[0i32; 3],
            &[f16::ZERO; 8],
            &[0i32; 1],
            3, // in_features
            8, // out_features
            2, // group_size — does not divide 3
        )
        .unwrap_err();
        assert!(matches!(err, DequantError::GroupSizeMisaligned { .. }));
    }

    /// Reference GPTQ dequant for `(i, j)` — single cell, used to
    /// hand-verify the formula in tests.
    fn gptq_reference_cell(
        w_int4: u8,
        z_int4: u8,
        scale: f32,
    ) -> f16 {
        let val = (w_int4 as i32 - (z_int4 as i32 + 1)) as f32 * scale;
        f16::from_f32(val)
    }

    #[test]
    fn gptq_dequant_single_block_matches_formula() {
        // in_features = 8 (1 in-pack), out_features = 8 (1 out-pack),
        // 1 group. g_idx = [0; 8]. Weights and zeros as specific
        // patterns so off-by-one is caught.
        let in_f = 8;
        let out_f = 8;
        // qweight: 1 in-pack × 8 outs. For column j, pack [j, j+1, ..., j+7].
        let qweight: Vec<i32> = (0..8u8)
            .map(|j| pack_int4_lane([j, j + 1, j + 2, j + 3, j + 4, j + 5, j + 6, j + 7]))
            .collect();
        // qzeros: 1 group × 1 out-pack, all lanes = 2.
        let qzeros = vec![pack_int4_lane([2; 8])];
        // scales: 1 group × 8 outs, all 0.5.
        let scales = vec![f16::from_f32(0.5); 8];
        let g_idx = vec![0i32; 8];

        let got = dequant_gptq_int4_to_f16(
            &qweight, &scales, &qzeros, &g_idx, in_f, out_f,
        )
        .unwrap();
        assert_eq!(got.len(), in_f * out_f);
        // For each (i, j): w_int4 = (j + i) since qweight col j has
        // lanes [j, j+1, ..., j+7]; z_int4 = 2; scale = 0.5
        for i in 0..in_f {
            for j in 0..out_f {
                let w_int4 = ((j as u8).wrapping_add(i as u8)) & 0xF;
                let exp = gptq_reference_cell(w_int4, 2, 0.5);
                let g = got[i * out_f + j];
                assert_eq!(
                    g.to_bits(),
                    exp.to_bits(),
                    "i={i}, j={j}: got {} vs exp {}",
                    g.to_f32(),
                    exp.to_f32(),
                );
            }
        }
    }

    #[test]
    fn gptq_dequant_honors_g_idx_remap() {
        // 16 in_features, 2 groups, but g_idx maps the first 8 to
        // group 1 and the last 8 to group 0 (a reversed actorder).
        // Pin that g_idx actually drives the group lookup; if the
        // dequant mistakenly used i/group_size we'd see different
        // numbers.
        let in_f = 16;
        let out_f = 8;
        let in_packs = in_f / 8;
        // qweight: every cell has int4 value = 5. Pack columns of
        // [5; 8] — same nibble in every lane.
        let qweight: Vec<i32> =
            (0..in_packs * out_f).map(|_| pack_int4_lane([5; 8])).collect();
        // qzeros: group 0 zeros = 1, group 1 zeros = 4. After +1
        // they become 2 and 5.
        let qzeros = vec![pack_int4_lane([1; 8]), pack_int4_lane([4; 8])];
        // scales: group 0 = 1.0, group 1 = 0.5
        let mut scales: Vec<f16> = Vec::new();
        scales.extend(std::iter::repeat(f16::from_f32(1.0)).take(out_f));
        scales.extend(std::iter::repeat(f16::from_f32(0.5)).take(out_f));
        // g_idx: reversed mapping.
        let mut g_idx = vec![1i32; 8];
        g_idx.extend(std::iter::repeat(0i32).take(8));

        let got = dequant_gptq_int4_to_f16(
            &qweight, &scales, &qzeros, &g_idx, in_f, out_f,
        )
        .unwrap();
        // Rows 0..8 (g=1): (5 - (4+1)) * 0.5 = 0.0
        for i in 0..8 {
            for j in 0..out_f {
                assert_eq!(
                    got[i * out_f + j].to_f32(),
                    0.0,
                    "row {i} (group 1) cell {j} should dequant to 0",
                );
            }
        }
        // Rows 8..16 (g=0): (5 - (1+1)) * 1.0 = 3.0
        for i in 8..16 {
            for j in 0..out_f {
                assert_eq!(
                    got[i * out_f + j].to_f32(),
                    3.0,
                    "row {i} (group 0) cell {j} should dequant to 3",
                );
            }
        }
    }

    #[test]
    fn gptq_dequant_rejects_out_of_range_g_idx() {
        // n_groups derives from scales.len() / out_features = 1.
        // g_idx[2] = 5 is well outside [0, 1).
        let in_f = 8;
        let out_f = 8;
        let qweight = vec![pack_int4_lane([0; 8]); out_f];
        let qzeros = vec![pack_int4_lane([0; 8])];
        let scales = vec![f16::from_f32(1.0); out_f];
        let mut g_idx = vec![0i32; in_f];
        g_idx[2] = 5; // out of range
        let err = dequant_gptq_int4_to_f16(
            &qweight, &scales, &qzeros, &g_idx, in_f, out_f,
        )
        .unwrap_err();
        match err {
            DequantError::GIdxOutOfRange { idx: 2, value: 5, n_groups: 1 } => {}
            other => panic!("expected GIdxOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn gptq_dequant_rejects_g_idx_shape_mismatch() {
        let in_f = 8;
        let out_f = 8;
        let qweight = vec![pack_int4_lane([0; 8]); out_f];
        let qzeros = vec![pack_int4_lane([0; 8])];
        let scales = vec![f16::from_f32(1.0); out_f];
        let g_idx = vec![0i32; 4]; // wrong size — should be in_f=8
        let err = dequant_gptq_int4_to_f16(
            &qweight, &scales, &qzeros, &g_idx, in_f, out_f,
        )
        .unwrap_err();
        match err {
            DequantError::GIdxShape { got: 4, in_features: 8 } => {}
            other => panic!("expected GIdxShape, got {other:?}"),
        }
    }

    #[test]
    fn gptq_dequant_rejects_unpacked_in_features() {
        // GPTQ packs along in_features; in_features must be a multiple of 8.
        let err = dequant_gptq_int4_to_f16(
            &[0i32; 0],
            &[f16::ZERO; 8],
            &[0i32; 1],
            &[0i32; 5], // g_idx (matches in_features = 5)
            5,          // in_features — not a multiple of 8
            8,
        )
        .unwrap_err();
        match err {
            DequantError::InFeaturesNotPacked { in_features: 5 } => {}
            other => panic!("expected InFeaturesNotPacked, got {other:?}"),
        }
    }
}
