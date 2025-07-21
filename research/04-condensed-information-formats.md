# Condensed Information Formats

## Overview

Condensed information formats extract and store the most relevant features from raw simulation data, enabling efficient visualization and analysis. This document explores various condensation techniques, indexing strategies, and progressive refinement approaches for massive simulation datasets.

## 1. Isosurface Extraction and Storage

### Overview

Isosurfaces represent surfaces of constant value in scalar fields, commonly used to visualize pressure boundaries, temperature fronts, or density interfaces in fluid simulations.

### Extraction Methods

#### Marching Cubes
```rust
struct MarchingCubesExtractor {
    // Grid data
    scalar_field: ScalarField3D,
    iso_value: f64,
    
    // Lookup tables
    edge_table: [u16; 256],
    tri_table: [[i8; 16]; 256],
    
    // Output mesh
    vertices: Vec<Vec3>,
    triangles: Vec<Triangle>,
    normals: Vec<Vec3>,
}

impl MarchingCubesExtractor {
    fn extract_isosurface(&mut self) -> Mesh {
        for each cube in grid {
            let cube_index = self.compute_cube_index(cube);
            let edge_mask = self.edge_table[cube_index];
            
            if edge_mask != 0 {
                let vertices = self.interpolate_vertices(cube, edge_mask);
                let triangles = self.generate_triangles(cube_index, vertices);
                self.add_to_mesh(triangles);
            }
        }
        self.build_mesh()
    }
}
```

#### Dual Contouring
```rust
struct DualContouringExtractor {
    // Hermite data
    scalar_field: ScalarField3D,
    gradient_field: VectorField3D,
    
    // Quadratic error functions
    qef_solver: QEFSolver,
    
    // Adaptive octree
    octree: AdaptiveOctree,
}

struct HermiteData {
    position: Vec3,
    normal: Vec3,
    value: f64,
}
```

#### Flying Edges (GPU-Optimized)
```rust
struct FlyingEdgesExtractor {
    // Three-pass algorithm
    pass1_edge_cases: Buffer<u32>,
    pass2_edge_counts: Buffer<u32>,
    pass3_vertices: Buffer<Vec3>,
    
    // GPU kernels
    classify_kernel: ComputeShader,
    compact_kernel: ComputeShader,
    generate_kernel: ComputeShader,
}
```

### Isosurface Storage Formats

#### Indexed Mesh Format
```rust
struct IsosurfaceMesh {
    // Mesh data
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
    normals: Vec<Vec3>,
    
    // Attributes
    scalar_values: Vec<f32>,      // Original field values
    gradient_magnitudes: Vec<f32>, // For feature detection
    
    // Metadata
    iso_value: f64,
    timestep: u64,
    extraction_method: ExtractionMethod,
    
    // Spatial index
    bvh: BoundingVolumeHierarchy,
}
```

#### Compressed Isosurface Format
```rust
struct CompressedIsosurface {
    // Quantized positions (16-bit per component)
    quantized_vertices: Vec<[u16; 3]>,
    quantization_box: AABB,
    
    // Delta-encoded connectivity
    delta_indices: Vec<i32>,
    
    // Compressed normals (octahedral encoding)
    encoded_normals: Vec<u16>,
    
    // Temporal compression
    base_frame: Option<IsosurfaceMesh>,
    delta_frames: Vec<DeltaFrame>,
}
```

### Multi-Resolution Isosurfaces

#### Hierarchical Isosurfaces
```rust
struct HierarchicalIsosurface {
    // Level of detail meshes
    lod_levels: Vec<LODLevel>,
    
    // Transition regions
    transition_cells: HashMap<(usize, usize), TransitionCell>,
    
    // Error metrics
    geometric_error: Vec<f32>,
    attribute_error: Vec<f32>,
}

struct LODLevel {
    resolution: [usize; 3],
    mesh: IsosurfaceMesh,
    simplification_error: f32,
    visible_range: (f32, f32),
}
```

## 2. Vorticity and Derived Quantities

### Vorticity Computation and Storage

```rust
struct VorticityData {
    // Vorticity vector field
    vorticity: VectorField3D,
    
    // Derived quantities
    vorticity_magnitude: ScalarField3D,
    helicity: ScalarField3D,  // v · ω
    enstrophy: ScalarField3D, // 0.5 * |ω|²
    
    // Vortex identification
    q_criterion: ScalarField3D,
    lambda2_criterion: ScalarField3D,
    delta_criterion: ScalarField3D,
}

impl VorticityData {
    fn compute_from_velocity(&mut self, velocity: &VectorField3D) {
        // Vorticity: ω = ∇ × v
        self.vorticity = velocity.curl();
        
        // Q-criterion: Q = 0.5(|Ω|² - |S|²)
        let velocity_gradient = velocity.gradient();
        let strain_rate = velocity_gradient.symmetric_part();
        let rotation_rate = velocity_gradient.antisymmetric_part();
        
        self.q_criterion = 0.5 * (rotation_rate.norm2() - strain_rate.norm2());
    }
}
```

### Vortex Core Extraction

```rust
struct VortexCore {
    // Core line representation
    centerline: Vec<Vec3>,
    
    // Core properties along line
    circulation: Vec<f64>,
    core_radius: Vec<f64>,
    axial_velocity: Vec<f64>,
    
    // Topology
    core_type: VortexType,
    connections: Vec<usize>,
}

enum VortexType {
    Primary,
    Secondary,
    Hairpin,
    Streamwise,
}

struct VortexCoreExtractor {
    // Parallel vectors method
    fn extract_parallel_vectors(&self, velocity: &VectorField3D, vorticity: &VectorField3D) -> Vec<VortexCore> {
        // Find where v || ω
        let parallel_regions = self.find_parallel_regions(velocity, vorticity);
        self.trace_core_lines(parallel_regions)
    }
    
    // Feature flow fields
    fn extract_feature_flow(&self, velocity_gradient: &TensorField3D) -> Vec<VortexCore> {
        let eigenvalues = velocity_gradient.eigenvalues();
        let critical_points = self.find_critical_points(eigenvalues);
        self.connect_critical_points(critical_points)
    }
}
```

### Feature-Based Compression

```rust
struct VortexBasedCompression {
    // Core vortex structures
    primary_vortices: Vec<VortexCore>,
    
    // Background flow
    background_field: CompressedField,
    
    // Reconstruction parameters
    vortex_models: Vec<VortexModel>,
}

struct VortexModel {
    model_type: VortexModelType,
    parameters: Vec<f64>,
    influence_radius: f64,
}

enum VortexModelType {
    RankineVortex,
    LambOseen,
    BurgerVortex,
    Sullivan,
}
```

## 3. Progressive Mesh Refinement

### Nanite-Inspired Architecture

```rust
struct NaniteStyleMesh {
    // Cluster hierarchy
    clusters: Vec<MeshCluster>,
    cluster_groups: Vec<ClusterGroup>,
    
    // LOD hierarchy
    lod_levels: Vec<LODData>,
    
    // Streaming data
    pages: Vec<StreamingPage>,
    page_dependencies: HashMap<PageId, Vec<PageId>>,
}

struct MeshCluster {
    // Geometry data
    vertices: [Vec3; 128],      // Fixed size for GPU efficiency
    triangles: [Triangle; 256], // Up to 256 triangles
    
    // Bounding info
    bounds: AABB,
    normal_cone: Cone,
    
    // LOD info
    parent_cluster: Option<ClusterId>,
    child_clusters: Vec<ClusterId>,
    error: f32,
}

struct StreamingPage {
    page_id: PageId,
    clusters: Vec<ClusterId>,
    compressed_data: Vec<u8>,
    size_bytes: usize,
}
```

### Meshlet Implementation

```rust
struct Meshlet {
    // Vertex data
    vertex_offset: u32,
    vertex_count: u8,
    
    // Triangle data
    triangle_offset: u32,
    triangle_count: u8,
    
    // Culling data
    bounding_sphere: Sphere,
    normal_cone: Cone,
    apex: Vec3,
}

struct MeshletMesh {
    // Global arrays
    vertices: Vec<Vertex>,
    vertex_indices: Vec<u8>,      // Local indices within meshlet
    triangle_indices: Vec<u32>,   // Global vertex indices
    
    // Meshlet data
    meshlets: Vec<Meshlet>,
    
    // Hierarchy
    meshlet_hierarchy: MeshletHierarchy,
}

struct MeshletHierarchy {
    levels: Vec<HierarchyLevel>,
    
    struct HierarchyLevel {
        meshlets: Vec<Meshlet>,
        parent_map: HashMap<MeshletId, MeshletId>,
        error_bounds: Vec<f32>,
    }
}
```

### Progressive Streaming Format

```rust
struct ProgressiveMeshFormat {
    // Header
    header: MeshHeader,
    
    // Base mesh (lowest LOD)
    base_mesh: CompressedMesh,
    
    // Progressive records
    refinement_records: Vec<RefinementRecord>,
    
    // Spatial index for selective loading
    spatial_index: OctreeIndex,
}

struct RefinementRecord {
    // Type of refinement
    operation: RefinementOp,
    
    // Affected region
    spatial_bounds: AABB,
    
    // Data
    vertex_data: Option<CompressedVertices>,
    connectivity_changes: Option<ConnectivityDelta>,
    
    // Dependencies
    required_records: Vec<RecordId>,
}

enum RefinementOp {
    VertexSplit,
    EdgeCollapse,
    FaceSubdivision,
    ClusterRefine,
}
```

## 4. Index-Based Storage Design

### Master Index Structure

```rust
struct SimulationDataIndex {
    // Metadata
    simulation_info: SimulationMetadata,
    
    // Temporal index
    time_index: TemporalIndex,
    
    // Spatial index
    spatial_index: SpatialIndex,
    
    // Feature index
    feature_index: FeatureIndex,
    
    // Data locations
    data_manifest: DataManifest,
}

struct TemporalIndex {
    // Time steps
    timesteps: Vec<f64>,
    
    // Key frames
    keyframes: Vec<KeyFrame>,
    
    // Temporal compression info
    compression_groups: Vec<TemporalGroup>,
}

struct SpatialIndex {
    // Octree nodes
    nodes: Vec<OctreeNode>,
    
    // Spatial hashing
    spatial_hash: HashMap<SpatialKey, Vec<DataChunk>>,
    
    // Multi-resolution levels
    resolution_levels: Vec<ResolutionLevel>,
}

struct FeatureIndex {
    // Feature catalog
    features: HashMap<FeatureType, Vec<FeatureInstance>>,
    
    // Feature tracking
    feature_tracks: Vec<FeatureTrack>,
    
    // Importance maps
    importance_fields: HashMap<String, ImportanceField>,
}
```

### S3-Compatible Storage Layout

```rust
struct S3StorageLayout {
    // Bucket structure
    bucket_name: String,
    
    // Key patterns
    index_key: String,              // "sim_001/index.json"
    metadata_prefix: String,        // "sim_001/metadata/"
    data_prefix: String,           // "sim_001/data/"
    
    // Chunk naming
    chunk_pattern: String,         // "{time}/{level}/{x}_{y}_{z}.chunk"
}

struct DataChunk {
    // Identification
    chunk_id: ChunkId,
    spatial_bounds: AABB,
    time_range: (f64, f64),
    
    // S3 location
    s3_key: String,
    byte_range: (u64, u64),
    
    // Compression
    compression: CompressionType,
    uncompressed_size: u64,
    compressed_size: u64,
    
    // Contents
    data_types: Vec<DataType>,
}
```

### Streaming Protocol

```rust
struct StreamingProtocol {
    // Priority queue for chunk requests
    request_queue: PriorityQueue<ChunkRequest>,
    
    // Active downloads
    active_downloads: HashMap<ChunkId, DownloadHandle>,
    
    // Cache management
    chunk_cache: LRUCache<ChunkId, ChunkData>,
    
    // Prefetching strategy
    prefetch_strategy: PrefetchStrategy,
}

struct ChunkRequest {
    chunk_id: ChunkId,
    priority: f32,
    deadline: Option<Instant>,
    quality_level: QualityLevel,
}

enum PrefetchStrategy {
    ViewFrustum { 
        look_ahead_distance: f32,
        fov_expansion: f32,
    },
    Temporal {
        future_frames: usize,
        past_frames: usize,
    },
    Feature {
        tracked_features: Vec<FeatureId>,
        influence_radius: f32,
    },
}
```

## 5. Common Condensed Formats

### Streamlines and Pathlines

```rust
struct StreamlineData {
    // Line geometry
    lines: Vec<Polyline>,
    
    // Attributes along lines
    velocity_magnitude: Vec<Vec<f32>>,
    pressure: Vec<Vec<f32>>,
    vorticity: Vec<Vec<f32>>,
    
    // Seeding information
    seed_points: Vec<Vec3>,
    integration_params: IntegrationParameters,
}

struct PathlineData {
    // Particle paths
    paths: Vec<TimePath>,
    
    // Time-varying attributes
    attributes: HashMap<String, Vec<TimeSeries>>,
    
    // Topology events
    birth_death_events: Vec<TopologyEvent>,
}
```

### Statistical Summaries

```rust
struct StatisticalSummary {
    // Spatial statistics
    mean_fields: HashMap<String, ScalarField3D>,
    variance_fields: HashMap<String, ScalarField3D>,
    
    // Temporal statistics
    time_averaged: HashMap<String, Field>,
    fluctuations: HashMap<String, Field>,
    
    // Spectral analysis
    power_spectra: HashMap<String, Spectrum>,
    coherent_structures: Vec<CoherentStructure>,
}

struct Spectrum {
    frequencies: Vec<f64>,
    power: Vec<f64>,
    phase: Vec<f64>,
}
```

### Feature Catalogs

```rust
struct FeatureCatalog {
    // Vortices
    vortex_cores: Vec<VortexCore>,
    vortex_sheets: Vec<VortexSheet>,
    
    // Shocks
    shock_surfaces: Vec<ShockSurface>,
    
    // Mixing regions
    mixing_layers: Vec<MixingLayer>,
    
    // Critical points
    stagnation_points: Vec<CriticalPoint>,
    separation_lines: Vec<SeparationLine>,
}
```

## 6. Compression Strategies

### Wavelet-Based Compression

```rust
struct WaveletCompression {
    // Wavelet transform
    wavelet_type: WaveletType,
    decomposition_levels: usize,
    
    // Coefficient storage
    significant_coefficients: HashMap<Level, SparseCoefficients>,
    
    // Error control
    absolute_tolerance: f64,
    relative_tolerance: f64,
}

struct SparseCoefficients {
    indices: Vec<CoeffIndex>,
    values: Vec<f64>,
    quantization_step: f64,
}
```

### Predictive Compression

```rust
struct PredictiveCompression {
    // Prediction model
    predictor: PredictionModel,
    
    // Residual storage
    residuals: CompressedResiduals,
    
    // Model parameters
    model_order: usize,
    adaptation_rate: f64,
}

enum PredictionModel {
    Linear { coefficients: Vec<f64> },
    Neural { network: NeuralPredictor },
    Physics { equations: PhysicsModel },
}
```

## 7. Metadata and Indexing

### Comprehensive Metadata

```rust
struct CondensedDataMetadata {
    // Source information
    source_simulation: SimulationReference,
    extraction_parameters: ExtractionParams,
    
    // Quality metrics
    approximation_error: QualityMetrics,
    
    // Relationships
    derived_from: Vec<DataId>,
    related_data: Vec<DataRelation>,
    
    // Validity
    valid_time_range: (f64, f64),
    valid_spatial_range: AABB,
}

struct QualityMetrics {
    max_error: f64,
    rms_error: f64,
    feature_preservation: f64,
    compression_ratio: f64,
}
```

### Search and Discovery

```rust
struct DataDiscovery {
    // Feature search
    fn find_features(&self, criteria: FeatureCriteria) -> Vec<FeatureRef> {
        self.feature_index.search(criteria)
    }
    
    // Spatial queries
    fn query_region(&self, region: AABB, time: f64) -> Vec<DataChunk> {
        self.spatial_index.query(region, time)
    }
    
    // Temporal queries
    fn query_timespan(&self, start: f64, end: f64) -> Vec<DataRef> {
        self.temporal_index.range_query(start, end)
    }
}
```

## Implementation Example

### Progressive Isosurface Viewer

```rust
struct ProgressiveIsosurfaceViewer {
    // Current view state
    camera: Camera,
    time: f64,
    
    // Loaded data
    base_mesh: Option<Mesh>,
    detail_patches: HashMap<PatchId, DetailPatch>,
    
    // Streaming state
    stream_manager: StreamManager,
    
    impl ProgressiveIsosurfaceViewer {
        fn update(&mut self) {
            // Determine required LOD
            let required_detail = self.compute_required_detail();
            
            // Request missing data
            for (region, lod) in required_detail {
                if !self.has_sufficient_detail(region, lod) {
                    self.stream_manager.request(region, lod);
                }
            }
            
            // Update mesh with available data
            self.update_progressive_mesh();
        }
        
        fn compute_required_detail(&self) -> Vec<(Region, LOD)> {
            // Screen-space error metric
            let mut requirements = Vec::new();
            
            for region in self.visible_regions() {
                let screen_size = self.project_to_screen(region);
                let required_lod = self.lod_from_screen_size(screen_size);
                requirements.push((region, required_lod));
            }
            
            requirements
        }
    }
}
```

## Best Practices

1. **Design for Progressive Loading**: Structure data for incremental refinement
2. **Implement Effective Indexing**: Enable fast spatial and temporal queries
3. **Use Appropriate Compression**: Balance quality vs. size for use case
4. **Maintain Relationships**: Track dependencies between derived data
5. **Enable Selective Access**: Support region-of-interest queries
6. **Optimize for Common Access Patterns**: Cache frequently accessed data
7. **Include Quality Metrics**: Store error bounds and confidence levels

## Future Directions

1. **AI-Driven Condensation**: Use ML to identify important features
2. **Semantic Compression**: Compress based on physical meaning
3. **Real-Time Extraction**: On-demand feature extraction during visualization
4. **Cloud-Native Formats**: Optimize for distributed storage and compute
5. **Standardization**: Develop common formats for tool interoperability

## References

1. "Isosurface Extraction in Time-varying Fields Using a Temporal Hierarchical Index Tree" - Shen et al.
2. "GPU-Based Tiled Ray Casting Using Depth Peeling" - Falk et al.
3. "Efficient Representation of Computational Meshes" - Nanite Documentation
4. "Progressive Meshes" - Hoppe
5. "Wavelet Compression of Time-Varying Volumetric Datasets" - Guthe & Strasser
6. "Feature-Based Volume Rendering of Time-Varying Data" - Silver & Wang