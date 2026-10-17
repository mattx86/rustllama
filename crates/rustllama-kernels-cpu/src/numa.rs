//! NUMA topology detection + node-local thread/memory placement (F11).
//!
//! On a multi-socket / multi-NUMA-node host a logical processor reaches its own
//! node's memory at full bandwidth but pays a cross-node penalty to reach a
//! sibling node's DIMMs. rustllama's default CPU pool is one global rayon pool
//! with no node awareness: a worker scheduled onto node 1 may stream weights
//! that physically live on node 0. This module detects the node topology and,
//! when the operator opts in (`[inference].numa_aware`) AND the host actually
//! has >= 2 nodes, groups the rayon workers by node (via
//! [`order_cpus_by_node`]) so consecutive workers share a node and touch
//! node-local memory — plus a node-local slab allocator ([`alloc_f32_slab_on_node`])
//! the weight/KV loaders can first-touch onto the owning node.
//!
//! **Single-node hosts — the overwhelming common case (every laptop/desktop and
//! this dev box) — are byte-identical to today**: [`detect`] reports one node,
//! the node-aware path is never taken, and the CPU-pool install falls straight
//! through to the historical `install_thread_pool{,_pinned}`. The node-aware
//! path is write-blind on PERFORMANCE (no 2-socket host here to A/B) but
//! byte-safe by construction — it only reorders which enabled CPU each worker
//! pins to; a 1-node topology makes it a no-op.
//!
//! Detection sources:
//! - **Linux** — `/sys/devices/system/node/node<N>/cpulist` (the kernel's own
//!   topology export; the `cpulist` format is parsed by [`parse_cpu_list`]).
//! - **Windows** — `GetNumaHighestNodeNumber` + `GetNumaNodeProcessorMask`
//!   (processor group 0; higher groups are left to the single-node fallback —
//!   the dev hardware is one group).
//! - **macOS / other** — no public per-node CPU map, so a single node covering
//!   every logical processor (correct: Apple Silicon + mainstream Intel Macs are
//!   UMA from software's point of view).

/// One NUMA node: a stable node id and the logical processors it owns
/// (ascending, de-duplicated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    pub id: u32,
    pub cpus: Vec<u32>,
}

/// The host's NUMA topology: an ordered list of nodes (ascending id). Always
/// has at least one node after [`detect`] (callers rely on `>= 1`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
}

impl NumaTopology {
    /// Number of distinct NUMA nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// `true` when the host has >= 2 nodes — the only case the node-aware path
    /// engages. A 1-node (or empty) topology returns `false`, keeping the CPU
    /// pool byte-identical to the historical behavior.
    pub fn is_multi_node(&self) -> bool {
        self.nodes.len() >= 2
    }

    /// The node id owning `cpu`, if any node claims it.
    pub fn cpu_node(&self, cpu: u32) -> Option<u32> {
        self.nodes
            .iter()
            .find(|n| n.cpus.contains(&cpu))
            .map(|n| n.id)
    }

    /// Single-node topology covering logical processors `0..n_cpus` on node 0.
    /// The universal fallback when no per-node map is available.
    pub fn single_node(n_cpus: u32) -> Self {
        Self {
            nodes: vec![NumaNode {
                id: 0,
                cpus: (0..n_cpus).collect(),
            }],
        }
    }

    /// Build from an explicit `(node_id, cpus)` list — the mock seam for tests
    /// (and any caller that already knows the map). Each node's CPUs are sorted
    /// + de-duplicated and nodes are ordered by id; empty nodes are dropped.
    pub fn from_nodes(nodes: Vec<NumaNode>) -> Self {
        let mut nodes: Vec<NumaNode> = nodes
            .into_iter()
            .filter_map(|mut n| {
                n.cpus.sort_unstable();
                n.cpus.dedup();
                (!n.cpus.is_empty()).then_some(n)
            })
            .collect();
        nodes.sort_by_key(|n| n.id);
        Self { nodes }
    }
}

/// Parse a Linux sysfs `cpulist` string (`"0-3,8,10-11"`) into a sorted,
/// de-duplicated CPU-index vector. Malformed fragments are skipped rather than
/// erroring — a best-effort perf hint, never a correctness input.
pub fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                if a <= b {
                    out.extend(a..=b);
                }
            }
        } else if let Ok(c) = part.parse::<u32>() {
            out.push(c);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Reorder `enabled` so all of node 0's enabled CPUs come first, then node 1's,
/// etc. (ascending node id; ascending CPU within a node). Enabled CPUs that no
/// node claims keep their input order at the tail. Pure + deterministic: the
/// node-aware pool pins worker `i` to `ordered[i % len]`, so a contiguous run of
/// workers lands on one node → node-local execution without a multi-pool
/// rewrite. For a single-node topology this returns `enabled` filtered through
/// node 0 (i.e. the same CPUs in ascending order).
pub fn order_cpus_by_node(enabled: &[u32], topo: &NumaTopology) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(enabled.len());
    for node in &topo.nodes {
        for &cpu in &node.cpus {
            if enabled.contains(&cpu) && !out.contains(&cpu) {
                out.push(cpu);
            }
        }
    }
    // Any enabled CPU not claimed by a node (mask gaps / hot-plug races): keep
    // it in the pool at its input position so no worker is dropped.
    for &cpu in enabled {
        if !out.contains(&cpu) {
            out.push(cpu);
        }
    }
    out
}

/// Detect the host NUMA topology. Always returns >= 1 node: a platform with no
/// per-node CPU map (or a detection failure) degrades to a single node spanning
/// every logical processor, which makes the node-aware path a safe no-op.
pub fn detect() -> NumaTopology {
    let topo = detect_impl();
    if topo.nodes.is_empty() {
        let n = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        NumaTopology::single_node(n.max(1))
    } else {
        topo
    }
}

#[cfg(target_os = "linux")]
fn detect_impl() -> NumaTopology {
    let base = std::path::Path::new("/sys/devices/system/node");
    let Ok(rd) = std::fs::read_dir(base) else {
        return NumaTopology::default();
    };
    let mut nodes = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Directories are `node0`, `node1`, ... — ignore `has_cpu`, `online`, etc.
        let Some(idstr) = name.strip_prefix("node") else {
            continue;
        };
        let Ok(id) = idstr.parse::<u32>() else {
            continue;
        };
        let Ok(list) = std::fs::read_to_string(entry.path().join("cpulist")) else {
            continue;
        };
        let cpus = parse_cpu_list(&list);
        if !cpus.is_empty() {
            nodes.push(NumaNode { id, cpus });
        }
    }
    nodes.sort_by_key(|n| n.id);
    NumaTopology { nodes }
}

#[cfg(windows)]
fn detect_impl() -> NumaTopology {
    extern "system" {
        fn GetNumaHighestNodeNumber(HighestNodeNumber: *mut u32) -> i32;
        fn GetNumaNodeProcessorMask(Node: u8, ProcessorMask: *mut u64) -> i32;
    }
    let mut highest: u32 = 0;
    // SAFETY: writes a single u32 through a valid pointer. A 0 return = failure.
    if unsafe { GetNumaHighestNodeNumber(&mut highest) } == 0 {
        return NumaTopology::default();
    }
    let mut nodes = Vec::new();
    for node in 0..=highest {
        // Node numbers above 255 can't be expressed through the legacy UCHAR
        // API; such systems are multi-group and handled by the fallback.
        if node > u8::MAX as u32 {
            break;
        }
        let mut mask: u64 = 0;
        // SAFETY: writes a single u64 through a valid pointer. The legacy mask
        // covers processor group 0 (the dev hardware is a single group).
        if unsafe { GetNumaNodeProcessorMask(node as u8, &mut mask) } == 0 {
            continue;
        }
        let cpus: Vec<u32> = (0..64u32).filter(|b| mask & (1u64 << b) != 0).collect();
        if !cpus.is_empty() {
            nodes.push(NumaNode { id: node, cpus });
        }
    }
    nodes.sort_by_key(|n| n.id);
    NumaTopology { nodes }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn detect_impl() -> NumaTopology {
    // macOS + other: no public per-node CPU map. The single-node fallback in
    // `detect` covers every logical processor (UMA from software's view).
    NumaTopology::default()
}

/// Allocate a zeroed `len`-element `f32` slab whose pages are first-touched by a
/// thread pinned to one of `node`'s CPUs, so Linux's default first-touch NUMA
/// policy places the physical pages on that node's memory. The weight/KV
/// loaders call this to keep a node's slab local to the workers pinned there.
///
/// Best-effort placement: on non-Linux, or when the node has no CPUs, it is a
/// plain zeroed allocation. The CONTENTS are identical on every platform — only
/// the physical page placement differs — so this never changes computed output.
pub fn alloc_f32_slab_on_node(len: usize, node: &NumaNode) -> Vec<f32> {
    #[cfg(target_os = "linux")]
    {
        if let Some(&cpu) = node.cpus.first() {
            // First-touch on a thread pinned to the node: the zero-write page
            // faults resolve on that node's memory. `thread::scope` keeps the
            // borrow of nothing simple and returns the slab by value.
            return std::thread::scope(|s| {
                s.spawn(move || {
                    super::pin_current_thread_to_cpu(cpu);
                    let mut v = vec![0.0f32; len];
                    // Touch one word per 4 KiB page to force the fault here
                    // (the allocator may hand back CoW zero pages otherwise).
                    let stride = (4096 / std::mem::size_of::<f32>()).max(1);
                    let mut i = 0;
                    while i < v.len() {
                        v[i] = 0.0;
                        i += stride;
                    }
                    v
                })
                .join()
                .unwrap_or_else(|_| vec![0.0f32; len])
            });
        }
    }
    let _ = node;
    vec![0.0f32; len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_ranges_singletons_and_gaps() {
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list(""), Vec::<u32>::new());
        assert_eq!(parse_cpu_list("  2-4 \n"), vec![2, 3, 4]);
        // Dups + reversed fragment are tolerated (reversed range dropped).
        assert_eq!(parse_cpu_list("1,1,2,4-2"), vec![1, 2]);
    }

    #[test]
    fn single_node_topology_is_not_multi_node() {
        let t = NumaTopology::single_node(8);
        assert_eq!(t.node_count(), 1);
        assert!(!t.is_multi_node());
        assert_eq!(t.cpu_node(3), Some(0));
        assert_eq!(t.cpu_node(99), None);
    }

    #[test]
    fn detect_always_returns_at_least_one_node() {
        // On this (single-node) box detection must never yield an empty topology.
        let t = detect();
        assert!(t.node_count() >= 1, "detect must return >= 1 node");
        assert!(!t.nodes[0].cpus.is_empty(), "node 0 must own >= 1 cpu");
    }

    #[test]
    fn from_nodes_normalizes_and_drops_empty() {
        let t = NumaTopology::from_nodes(vec![
            NumaNode {
                id: 1,
                cpus: vec![3, 2, 2],
            },
            NumaNode {
                id: 0,
                cpus: vec![1, 0],
            },
            NumaNode {
                id: 2,
                cpus: vec![],
            },
        ]);
        // Reordered by id, deduped, empty node 2 dropped.
        assert_eq!(t.node_count(), 2);
        assert_eq!(t.nodes[0].id, 0);
        assert_eq!(t.nodes[0].cpus, vec![0, 1]);
        assert_eq!(t.nodes[1].id, 1);
        assert_eq!(t.nodes[1].cpus, vec![2, 3]);
    }

    #[test]
    fn order_cpus_by_node_groups_by_node_then_cpu() {
        // Mock 2-socket topology: node 0 = {0,1,2,3}, node 1 = {4,5,6,7}.
        let topo = NumaTopology::from_nodes(vec![
            NumaNode {
                id: 0,
                cpus: vec![0, 1, 2, 3],
            },
            NumaNode {
                id: 1,
                cpus: vec![4, 5, 6, 7],
            },
        ]);
        assert!(topo.is_multi_node());
        // An interleaved enabled set must come out node-grouped.
        let enabled = vec![5, 0, 6, 1, 7, 2];
        let ordered = order_cpus_by_node(&enabled, &topo);
        assert_eq!(ordered, vec![0, 1, 2, 5, 6, 7]);
        // Worker i pins to ordered[i % len]: the first 3 workers all land on
        // node 0, the next 3 on node 1 — node-local by construction.
        assert_eq!(topo.cpu_node(ordered[0]), Some(0));
        assert_eq!(topo.cpu_node(ordered[2]), Some(0));
        assert_eq!(topo.cpu_node(ordered[3]), Some(1));
    }

    #[test]
    fn order_cpus_by_node_keeps_unclaimed_cpus() {
        let topo = NumaTopology::from_nodes(vec![NumaNode {
            id: 0,
            cpus: vec![0, 1],
        }]);
        // CPU 9 belongs to no node → kept at the tail, nothing dropped.
        let ordered = order_cpus_by_node(&[9, 1, 0], &topo);
        assert_eq!(ordered, vec![0, 1, 9]);
    }

    #[test]
    fn order_cpus_by_node_single_node_is_ascending_passthrough() {
        // The byte-identical guard: single-node ordering == the enabled set in
        // ascending order (what the CLI already passes), so delegating to the
        // pinned installer changes nothing.
        let topo = NumaTopology::single_node(8);
        let enabled: Vec<u32> = (0..8).collect();
        assert_eq!(order_cpus_by_node(&enabled, &topo), enabled);
    }

    #[test]
    fn alloc_slab_contents_are_zeroed_regardless_of_placement() {
        let node = NumaNode {
            id: 0,
            cpus: vec![0],
        };
        let slab = alloc_f32_slab_on_node(10_000, &node);
        assert_eq!(slab.len(), 10_000);
        assert!(slab.iter().all(|&x| x == 0.0), "slab must be zeroed");
        // Empty-node path (no CPU to pin) still yields correct contents.
        let empty = NumaNode {
            id: 7,
            cpus: vec![],
        };
        let slab2 = alloc_f32_slab_on_node(16, &empty);
        assert_eq!(slab2, vec![0.0f32; 16]);
    }
}
