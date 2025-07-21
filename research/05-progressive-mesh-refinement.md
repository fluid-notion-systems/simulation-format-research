# Progressive Mesh Refinement

## Overview

Progressive mesh refinement enables efficient visualization of massive simulation datasets by providing multiple levels of detail (LOD) that can be streamed and refined based on viewing conditions. This document explores modern approaches, with particular focus on Nanite-like virtualized geometry and Bevy's meshlet implementation.

## 1. Fundamentals of Progressive Meshes

### Core Concepts

Progressive meshes represent 3D geometry as a coarse base mesh plus a sequence of refinement operations that incrementally add detail.

```rust
struct ProgressiveMesh {
    // Base mesh (coarsest representation)
    base_vertices: Vec<Vec3>,
    base_triangles: Vec<Triangle>,
    
    // Refinement operations
    vertex_splits: Vec<VertexSplit>,
    
    // Current state
    current_vertices: Vec<Vec3>,
    current_triangles: Vec<Triangle>,
    current_lod: usize,
}

struct VertexSplit {
    // Parent vertex to split
    parent_vertex: VertexId,
    
    // New vertices created
    child_vertices: (VertexId, VertexId),
    
    // Affected triangles
    removed_triangles: Vec<TriangleId>,
    added_triangles: Vec<Triangle>,
    
    // Error metric
    geometric_error: f32,
}
```

### Edge Collapse and Vertex Split

The fundamental operations for progressive mesh construction:

```rust
impl ProgressiveMesh {
    fn edge_collapse(&mut self, edge: EdgeId) -> EdgeCollapse {
        // Collapse edge to single vertex
        let (v1, v2) = self.get_edge_vertices(edge);
        let new_vertex = self.compute_optimal_position(v1, v2);
        
        // Record operation for reversal
        EdgeCollapse {
            collapsed_edge: edge,
            new_vertex_position: new_vertex,
            affected_triangles: self.get_adjacent_triangles(edge),
        }
    }
    
    fn vertex_split(&mut self, split: &VertexSplit) {
        // Reverse of edge collapse
        self.current_vertices[split.parent_vertex] = split.child_vertices.0;
        self.current_vertices.push(split.child_vertices.1);
        
        // Update connectivity
        for tri in &split.removed_triangles {
            self.current_triangles.remove(tri);
        }
        for tri in &split.added_triangles {
            self.current_triangles.push(*tri);
        }
    }
}
```

## 2. Nanite-Style Virtualized Geometry

### Architecture Overview

Nanite uses a hierarchical cluster-based approach where geometry is divided into small clusters that can be efficiently culled and rendered.

```rust
struct NaniteGeometry {
    // Cluster hierarchy
    clusters: Vec<GeometryCluster>,
    cluster_hierarchy: ClusterDAG,
    
    // Streaming pages
    pages: Vec<GeometryPage>,
    page_table: PageTable,
    
    // Runtime state
    visible_clusters: BitSet,
    loaded_pages: LRUCache<PageId, GeometryPage>,
}

struct GeometryCluster {
    // Fixed-size geometry data
    vertices: [Vertex; MAX_CLUSTER_VERTS],  // Usually 128
    triangles: [Triangle; MAX_CLUSTER_TRIS], // Usually 256
    vertex_count: u8,
    triangle_count: u8,
    
    // Bounding volumes
    bounding_sphere: Sphere,
    normal_cone: Cone,
    
    // LOD information
    parent_cluster: Option<ClusterId>,
    child_clusters: Vec<ClusterId>,
    lod_error: f32,
    
    // Visibility data
    occlusion_bounds: AABB,
}
```

### Cluster Building Algorithm

```rust
struct ClusterBuilder {
    target_triangle_count: usize,  // e.g., 128
    
    fn build_clusters(&self, mesh: &Mesh) -> Vec<GeometryCluster> {
        let mut clusters = Vec::new();
        let mut remaining_triangles = mesh.triangles.clone();
        
        while !remaining_triangles.is_empty() {
            // Grow cluster from seed
            let seed_triangle = self.pick_seed_triangle(&remaining_triangles);
            let cluster = self.grow_cluster(seed_triangle, &mut remaining_triangles);
            clusters.push(cluster);
        }
        
        clusters
    }
    
    fn grow_cluster(&self, seed: TriangleId, triangles: &mut Vec<Triangle>) -> GeometryCluster {
        let mut cluster_triangles = vec![seed];
        let mut boundary = self.get_adjacent_triangles(seed);
        
        while cluster_triangles.len() < self.target_triangle_count && !boundary.is_empty() {
            // Pick best triangle to add
            let best_triangle = self.pick_best_triangle(&boundary, &cluster_triangles);
            cluster_triangles.push(best_triangle);
            
            // Update boundary
            self.update_boundary(&mut boundary, best_triangle);
        }
        
        self.create_cluster(cluster_triangles)
    }
}
```

### Hierarchical LOD Construction

```rust
struct LODBuilder {
    simplification_method: SimplificationMethod,
    
    fn build_lod_hierarchy(&self, clusters: Vec<GeometryCluster>) -> ClusterDAG {
        let mut hierarchy = ClusterDAG::new();
        let mut current_level = clusters;
        
        while current_level.len() > 1 {
            let parent_level = self.create_parent_level(&current_level);
            
            // Link parent-child relationships
            for (parent_idx, parent) in parent_level.iter().enumerate() {
                for child_idx in parent.source_clusters {
                    hierarchy.add_edge(parent_idx, child_idx);
                }
            }
            
            current_level = parent_level;
        }
        
        hierarchy
    }
    
    fn create_parent_level(&self, clusters: &[GeometryCluster]) -> Vec<GeometryCluster> {
        // Group spatially close clusters
        let groups = self.group_clusters(clusters);
        
        groups.into_iter()
            .map(|group| self.simplify_cluster_group(group))
            .collect()
    }
}
```

### GPU-Driven Culling

```rust
struct GPUCulling {
    // Culling compute shaders
    frustum_cull_shader: ComputeShader,
    occlusion_cull_shader: ComputeShader,
    cluster_cull_shader: ComputeShader,
    
    // GPU buffers
    cluster_buffer: Buffer<GeometryCluster>,
    visibility_buffer: Buffer<u32>,
    
    fn execute_culling(&self, view: &ViewData) -> VisibleClusters {
        // Two-phase culling
        
        // Phase 1: Cull cluster groups
        self.frustum_cull_shader.dispatch(
            self.cluster_buffer,
            view.frustum,
            self.visibility_buffer
        );
        
        // Phase 2: Fine-grained cluster culling
        self.cluster_cull_shader.dispatch(
            self.visible_groups(),
            view,
            self.visibility_buffer
        );
        
        // Read back results
        self.visibility_buffer.read()
    }
}
```

## 3. Bevy's Meshlet Implementation

### Meshlet Structure

Bevy's meshlet system divides meshes into small, GPU-friendly chunks optimized for modern rendering pipelines.

```rust
// Based on Bevy's meshlet design
#[repr(C)]
struct Meshlet {
    // Vertex data
    vertex_offset: u32,
    vertex_count: u8,
    
    // Triangle data  
    triangle_offset: u32,
    triangle_count: u8,
    
    // Culling data
    bounding_sphere: [f32; 4],  // xyz = center, w = radius
    
    // Simplified normal cone for backface culling
    // xyz = axis, w = -cos(angle)
    cone_axis_cutoff: [f32; 4],
}

struct MeshletMesh {
    // Vertex buffer
    vertex_buffer: Buffer<Vertex>,
    
    // Index buffer (8-bit local indices)
    meshlet_vertex_indices: Buffer<u8>,
    
    // Triangle indices (pointing to vertex buffer)
    meshlet_triangle_indices: Buffer<u32>,
    
    // Meshlet data
    meshlets: Buffer<Meshlet>,
    
    // LOD chain
    lod_meshlet_counts: Vec<u32>,
}
```

### Meshlet Building Process

```rust
impl MeshletBuilder {
    const MAX_VERTICES: usize = 64;
    const MAX_TRIANGLES: usize = 124;  // Optimized for 512-byte meshlets
    
    fn build_meshlets(&self, mesh: &Mesh) -> MeshletMesh {
        let mut meshlets = Vec::new();
        let mut processed_triangles = BitSet::new(mesh.triangles.len());
        
        while processed_triangles.count() < mesh.triangles.len() {
            let seed = self.find_unprocessed_triangle(&processed_triangles);
            let meshlet = self.grow_meshlet(mesh, seed, &mut processed_triangles);
            meshlets.push(meshlet);
        }
        
        self.pack_meshlet_data(meshlets)
    }
    
    fn grow_meshlet(&self, mesh: &Mesh, seed: usize, processed: &mut BitSet) -> Meshlet {
        let mut meshlet_verts = IndexSet::new();
        let mut meshlet_tris = Vec::new();
        
        // Add seed triangle
        let seed_tri = &mesh.triangles[seed];
        meshlet_verts.extend(&[seed_tri.v0, seed_tri.v1, seed_tri.v2]);
        meshlet_tris.push(seed);
        processed.set(seed);
        
        // Grow meshlet
        let mut candidates = self.get_adjacent_triangles(mesh, seed);
        
        while meshlet_verts.len() < Self::MAX_VERTICES && 
              meshlet_tris.len() < Self::MAX_TRIANGLES &&
              !candidates.is_empty() {
            
            // Score candidates
            let best = candidates.iter()
                .filter(|&&tri| !processed.get(tri))
                .filter(|&&tri| self.can_add_triangle(mesh, tri, &meshlet_verts))
                .min_by_key(|&&tri| self.score_triangle(mesh, tri, &meshlet_verts))
                .copied();
            
            if let Some(tri_idx) = best {
                let tri = &mesh.triangles[tri_idx];
                meshlet_verts.extend(&[tri.v0, tri.v1, tri.v2]);
                meshlet_tris.push(tri_idx);
                processed.set(tri_idx);
                
                // Update candidates
                candidates.extend(self.get_adjacent_triangles(mesh, tri_idx));
            } else {
                break;
            }
        }
        
        self.finalize_meshlet(mesh, meshlet_verts, meshlet_tris)
    }
}
```

### GPU Rendering Pipeline

```rust
struct MeshletRenderer {
    // Pipeline stages
    task_shader: TaskShader,
    mesh_shader: MeshShader,
    fragment_shader: FragmentShader,
    
    fn render(&self, meshlet_mesh: &MeshletMesh, view: &View) {
        // Task shader: Cull meshlets and emit mesh shader workgroups
        let visible_meshlets = self.task_shader.cull_meshlets(
            meshlet_mesh,
            view.frustum,
            view.occlusion_buffer
        );
        
        // Mesh shader: Process visible meshlets
        for meshlet_id in visible_meshlets {
            self.mesh_shader.process_meshlet(
                meshlet_mesh,
                meshlet_id,
                view.transform
            );
        }
        
        // Fragment shader: Standard shading
        self.fragment_shader.shade();
    }
}
```

## 4. Streaming and Memory Management

### Page-Based Streaming System

```rust
struct StreamingSystem {
    // Page management
    page_size: usize,  // e.g., 64KB
    page_table: HashMap<PageId, PageLocation>,
    
    // Memory pools
    gpu_pool: GPUMemoryPool,
    cpu_cache: LRUCache<PageId, Page>,
    
    // Streaming state
    active_requests: HashMap<PageId, RequestHandle>,
    request_queue: PriorityQueue<PageRequest>,
}

struct Page {
    page_id: PageId,
    compression: CompressionType,
    data: Vec<u8>,
    
    // Content description
    cluster_range: Range<ClusterId>,
    vertex_data_offset: usize,
    index_data_offset: usize,
}

impl StreamingSystem {
    fn request_page(&mut self, page_id: PageId, priority: f32) {
        if !self.is_loaded(page_id) && !self.is_pending(page_id) {
            let request = PageRequest {
                page_id,
                priority,
                timestamp: Instant::now(),
            };
            self.request_queue.push(request);
        }
    }
    
    fn update(&mut self) {
        // Process completed requests
        self.handle_completed_requests();
        
        // Start new requests
        while self.active_requests.len() < MAX_CONCURRENT_REQUESTS {
            if let Some(request) = self.request_queue.pop() {
                self.start_request(request);
            } else {
                break;
            }
        }
        
        // Evict old pages if needed
        self.evict_pages_if_needed();
    }
}
```

### Predictive Loading

```rust
struct PredictiveLoader {
    // Movement prediction
    velocity_estimator: VelocityEstimator,
    
    // View analysis
    view_analyzer: ViewAnalyzer,
    
    fn predict_required_pages(&self, current_view: &View, dt: f32) -> Vec<PageId> {
        let mut required_pages = Vec::new();
        
        // Predict future camera position
        let predicted_view = self.velocity_estimator.predict(current_view, dt);
        
        // Find potentially visible clusters
        let visible_bounds = self.view_analyzer.compute_visible_bounds(&predicted_view);
        
        // Map to required pages
        for cluster_id in self.find_clusters_in_bounds(visible_bounds) {
            let page_id = self.cluster_to_page(cluster_id);
            required_pages.push(page_id);
        }
        
        required_pages
    }
}
```

## 5. Error Metrics and LOD Selection

### Screen-Space Error

```rust
struct ScreenSpaceError {
    fn compute_error(&self, cluster: &GeometryCluster, view: &View) -> f32 {
        // Project bounding sphere to screen
        let center_view = view.world_to_view * cluster.bounding_sphere.center;
        let distance = center_view.z;
        
        // Compute projected radius
        let angular_size = cluster.bounding_sphere.radius / distance;
        let screen_radius = angular_size * view.viewport_height * 0.5 / view.fov_y.tan();
        
        // Error metric: geometric deviation / screen size
        cluster.lod_error / screen_radius
    }
    
    fn select_lod(&self, cluster_dag: &ClusterDAG, view: &View, error_threshold: f32) -> Vec<ClusterId> {
        let mut selected = Vec::new();
        let mut queue = vec![cluster_dag.root()];
        
        while let Some(cluster_id) = queue.pop() {
            let cluster = &cluster_dag[cluster_id];
            let error = self.compute_error(cluster, view);
            
            if error < error_threshold || cluster.is_leaf() {
                selected.push(cluster_id);
            } else {
                // Refine: use children instead
                queue.extend(&cluster.child_clusters);
            }
        }
        
        selected
    }
}
```

### Temporal Coherence

```rust
struct TemporalLOD {
    // Previous frame state
    previous_selection: HashMap<ClusterId, LODLevel>,
    
    // Hysteresis thresholds
    refine_threshold: f32,
    coarsen_threshold: f32,
    
    fn update_lod_selection(&mut self, view: &View) -> Vec<ClusterId> {
        let mut new_selection = HashMap::new();
        
        for (&cluster_id, &prev_lod) in &self.previous_selection {
            let error = self.compute_error(cluster_id, view);
            
            // Apply hysteresis
            let new_lod = if error > self.refine_threshold {
                prev_lod + 1  // Refine
            } else if error < self.coarsen_threshold {
                prev_lod.saturating_sub(1)  // Coarsen
            } else {
                prev_lod  // Keep same
            };
            
            new_selection.insert(cluster_id, new_lod);
        }
        
        self.previous_selection = new_selection;
        self.lod_to_clusters()
    }
}
```

## 6. Integration with Simulation Data

### Time-Varying Meshes

```rust
struct ProgressiveSimulationMesh {
    // Base topology (may change over time)
    topology_keyframes: Vec<(f64, MeshletMesh)>,
    
    // Deformation data
    vertex_displacements: TimeVaryingField<Vec3>,
    
    // Attribute data
    scalar_attributes: HashMap<String, TimeVaryingField<f32>>,
    
    fn get_mesh_at_time(&self, time: f64, lod: LODLevel) -> MeshletMesh {
        // Find topology keyframes
        let (t0, t1, alpha) = self.find_keyframe_bounds(time);
        
        // Interpolate or use nearest
        let base_mesh = if self.topology_changes_between(t0, t1) {
            &self.topology_keyframes[t0].1  // Use nearest
        } else {
            self.interpolate_mesh(t0, t1, alpha)  // Interpolate
        };
        
        // Apply LOD selection
        self.select_lod_meshlets(base_mesh, lod)
    }
}
```

### Adaptive Refinement Based on Data

```rust
struct DataDrivenRefinement {
    // Refinement criteria
    criteria: Vec<RefinementCriterion>,
    
    fn compute_refinement_priority(&self, cluster: &GeometryCluster, data: &SimulationData) -> f32 {
        let mut priority = 0.0;
        
        for criterion in &self.criteria {
            match criterion {
                RefinementCriterion::VorticityMagnitude { threshold } => {
                    let avg_vorticity = self.average_in_bounds(
                        &data.vorticity,
                        &cluster.bounding_sphere
                    );
                    if avg_vorticity > *threshold {
                        priority += avg_vorticity / threshold;
                    }
                }
                RefinementCriterion::PressureGradient { threshold } => {
                    let max_gradient = self.max_in_bounds(
                        &data.pressure_gradient,
                        &cluster.bounding_sphere
                    );
                    if max_gradient > *threshold {
                        priority += max_gradient / threshold;
                    }
                }
                // Other criteria...
            }
        }
        
        priority
    }
}
```

## 7. Performance Optimizations

### GPU Memory Management

```rust
struct GPUMemoryManager {
    // Memory heaps
    vertex_heap: GPUHeap,
    index_heap: GPUHeap,
    meshlet_heap: GPUHeap,
    
    // Allocation tracking
    allocations: HashMap<MeshId, GPUAllocation>,
    
    // Defragmentation
    defrag_threshold: f32,
    
    fn allocate_meshlet_mesh(&mut self, mesh: &MeshletMesh) -> Result<GPUAllocation, AllocError> {
        let vertex_size = mesh.vertex_count() * size_of::<Vertex>();
        let index_size = mesh.index_count() * size_of::<u32>();
        let meshlet_size = mesh.meshlet_count() * size_of::<Meshlet>();
        
        // Try to allocate contiguously
        if let Ok(allocation) = self.allocate_contiguous(vertex_size + index_size + meshlet_size) {
            Ok(allocation)
        } else {
            // Fall back to separate allocations
            self.allocate_separate(vertex_size, index_size, meshlet_size)
        }
    }
    
    fn defragment(&mut self) {
        if self.fragmentation_ratio() > self.defrag_threshold {
            // Compact allocations
            self.compact_heaps();
        }
    }
}
```

### Batching and Instancing

```rust
struct MeshletBatcher {
    // Batch buckets by material and LOD
    batches: HashMap<(MaterialId, LODLevel), Batch>,
    
    fn add_instance(&mut self, instance: MeshletInstance) {
        let key = (instance.material_id, instance.lod_level);
        self.batches.entry(key)
            .or_insert_with(Batch::new)
            .add(instance);
    }
    
    fn generate_draw_commands(&self) -> Vec<DrawCommand> {
        self.batches.values()
            .filter(|batch| batch.instance_count() > 0)
            .map(|batch| batch.to_draw_command())
            .collect()
    }
}
```

## 8. Quality Metrics and Validation

### Mesh Quality Preservation

```rust
struct QualityMetrics {
    fn measure_simplification_error(&self, original: &Mesh, simplified: &Mesh) -> ErrorMetrics {
        ErrorMetrics {
            hausdorff_distance: self.compute_hausdorff(original, simplified),
            mean_squared_error: self.compute_mse(original, simplified),
            normal_deviation: self.compute_normal_deviation(original, simplified),
            volume_change: self.compute_volume_change(original, simplified),
        }
    }
    
    fn validate_progressive_mesh(&self, pm: &ProgressiveMesh) -> ValidationResult {
        let mut results = Vec::new();
        
        // Test each LOD level
        for lod in 0..pm.num_lods() {
            let mesh = pm.get_lod(lod);
            
            // Check mesh validity
            if let Err(e) = self.validate_mesh_topology(&mesh) {
                results.push(ValidationError::InvalidTopology(lod, e));
            }
            
            // Check error bounds
            if lod > 0 {
                let prev_mesh = pm.get_lod(lod - 1);
                let error = self.measure_simplification_error(&prev_mesh, &mesh);
                if error.exceeds_threshold(pm.get_error_threshold(lod)) {
                    results.push(ValidationError::ExcessiveError(lod, error));
                }
            }
        }
        
        ValidationResult { errors: results }
    }
}
```

## Best Practices

1. **Cluster Size Optimization**: Balance between GPU efficiency and culling granularity
2. **Error Metric Selection**: Choose appropriate metrics for your data type
3. **Memory Budgeting**: Implement strict memory limits with graceful degradation
4. **Temporal Coherence**: Minimize LOD changes between frames
5. **Compression Integration**: Compress both geometry and attributes
6. **Profile-Guided Optimization**: Use profiling data to tune parameters

## Future Directions

1. **Neural Mesh Compression**: Use learned representations for extreme compression
2. **Ray Tracing Integration**: Adapt progressive meshes for RT acceleration structures
3. **Procedural Refinement**: Generate detail procedurally instead of storing
4. **Topology Adaptation**: Handle changing mesh connectivity over time
5. **Multi-Resolution Physics**: Couple LOD with simulation accuracy

## References

1. "Nanite: A Deep Dive" - Epic Games Documentation
2. "Meshlets: Quadric-based Mesh Simplification" - Garland & Heckbert
3. "Progressive Meshes" - Hugues Hoppe
4. "GPU-Driven Rendering Pipelines" - Wihlidal
5. "Bevy Meshlet Implementation" - Bevy Engine Source Code
6. "Geometry Images" - Gu et al.
7. "Streaming Compressed 3D Data on the Web" - Limper et al.