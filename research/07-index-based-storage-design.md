# Index-Based Storage Design

## Overview

Index-based storage design is crucial for efficiently managing and accessing massive simulation datasets. This document explores comprehensive indexing strategies that enable rapid data discovery, progressive loading, and cloud-native storage patterns optimized for S3 and similar object stores.

## 1. Core Index Architecture

### Hierarchical Index Structure

```rust
pub struct SimulationIndex {
    // Master index metadata
    metadata: IndexMetadata,
    
    // Temporal indexing
    temporal_index: TemporalIndex,
    
    // Spatial indexing
    spatial_index: SpatialIndex,
    
    // Feature-based indexing
    feature_index: FeatureIndex,
    
    // Data manifest
    data_manifest: DataManifest,
}

pub struct IndexMetadata {
    // Simulation identification
    simulation_id: Uuid,
    simulation_type: SimulationType,
    creation_timestamp: DateTime<Utc>,
    
    // Physical parameters
    domain_bounds: AABB,
    time_range: (f64, f64),
    
    // Data characteristics
    total_size_bytes: u64,
    compression_ratio: f32,
    chunk_size_bytes: u64,
    
    // Index version and schema
    index_version: Version,
    schema_version: Version,
}
```

### Multi-Level Index Design

```rust
pub struct MultiLevelIndex {
    // Level 0: Coarse overview
    overview_index: OverviewIndex,
    
    // Level 1: Region index
    region_index: HashMap<RegionId, RegionIndex>,
    
    // Level 2: Chunk index
    chunk_index: HashMap<ChunkId, ChunkIndex>,
    
    // Level 3: Fine-grained index
    detail_index: Option<DetailIndex>,
}

pub struct OverviewIndex {
    // Temporal summary
    time_keyframes: Vec<TimeKeyframe>,
    
    // Spatial summary
    spatial_tree: OctreeIndex,
    
    // Feature summary
    major_features: Vec<FeatureSummary>,
    
    // Quick statistics
    global_stats: GlobalStatistics,
}
```

## 2. Temporal Indexing

### Time-Based Organization

```rust
pub struct TemporalIndex {
    // Time series organization
    timesteps: Vec<Timestep>,
    
    // Keyframe indices
    keyframes: Vec<KeyframeIndex>,
    
    // Delta compression groups
    compression_groups: Vec<CompressionGroup>,
    
    // Temporal relationships
    temporal_graph: TemporalGraph,
}

pub struct Timestep {
    time: f64,
    global_index: u64,
    
    // Data location
    chunk_refs: Vec<ChunkReference>,
    
    // Quick access metadata
    bounds: AABB,
    feature_flags: FeatureFlags,
    quality_metrics: QualityMetrics,
}

pub struct KeyframeIndex {
    time: f64,
    
    // Full data reference
    full_data_ref: DataReference,
    
    // Dependent frames
    dependent_frames: Range<u64>,
    
    // Compression base
    is_i_frame: bool,
}

impl TemporalIndex {
    pub fn find_time_range(&self, start: f64, end: f64) -> Vec<ChunkReference> {
        let start_idx = self.timesteps.binary_search_by(|t| 
            t.time.partial_cmp(&start).unwrap()
        ).unwrap_or_else(|i| i);
        
        let end_idx = self.timesteps.binary_search_by(|t| 
            t.time.partial_cmp(&end).unwrap()
        ).unwrap_or_else(|i| i);
        
        self.timesteps[start_idx..end_idx]
            .iter()
            .flat_map(|t| t.chunk_refs.clone())
            .collect()
    }
}
```

### Temporal Compression Groups

```rust
pub struct CompressionGroup {
    // Group identification
    group_id: GroupId,
    time_range: (f64, f64),
    
    // Base frame
    base_frame: ChunkReference,
    
    // Delta frames
    delta_frames: Vec<DeltaFrame>,
    
    // Group statistics
    compression_ratio: f32,
    max_error: f32,
}

pub struct DeltaFrame {
    time: f64,
    delta_size: u64,
    
    // Reference to delta data
    delta_ref: ChunkReference,
    
    // Dependencies
    depends_on: Vec<FrameId>,
}
```

## 3. Spatial Indexing

### Octree-Based Spatial Index

```rust
pub struct SpatialIndex {
    // Root octree node
    root: OctreeNode,
    
    // Level of detail mapping
    lod_levels: Vec<LODLevel>,
    
    // Spatial hashing for fast lookup
    spatial_hash: HashMap<SpatialHash, Vec<ChunkId>>,
    
    // Importance map
    importance_field: ImportanceField,
}

pub struct OctreeNode {
    // Node identification
    node_id: NodeId,
    bounds: AABB,
    level: u8,
    
    // Children (if internal node)
    children: Option<[Box<OctreeNode>; 8]>,
    
    // Data (if leaf node)
    chunks: Vec<ChunkReference>,
    
    // Metadata
    total_points: u64,
    avg_density: f32,
    contains_features: FeatureFlags,
}

impl SpatialIndex {
    pub fn query_region(&self, region: &AABB) -> Vec<ChunkReference> {
        let mut results = Vec::new();
        self.query_recursive(&self.root, region, &mut results);
        results
    }
    
    fn query_recursive(&self, node: &OctreeNode, region: &AABB, results: &mut Vec<ChunkReference>) {
        if !node.bounds.intersects(region) {
            return;
        }
        
        if let Some(ref children) = node.children {
            // Internal node - recurse
            for child in children.iter() {
                self.query_recursive(child, region, results);
            }
        } else {
            // Leaf node - add chunks
            results.extend(node.chunks.clone());
        }
    }
}
```

### Hierarchical Spatial Hashing

```rust
pub struct HierarchicalSpatialHash {
    // Multiple resolution levels
    levels: Vec<SpatialHashLevel>,
    
    // Adaptive grid parameters
    base_cell_size: f32,
    max_levels: u8,
}

pub struct SpatialHashLevel {
    level: u8,
    cell_size: f32,
    
    // Hash table
    hash_table: HashMap<SpatialKey, CellData>,
    
    // Occupancy grid for fast empty space skipping
    occupancy: BitGrid3D,
}

#[derive(Hash, Eq, PartialEq)]
pub struct SpatialKey {
    x: i32,
    y: i32,
    z: i32,
    level: u8,
}
```

## 4. Feature-Based Indexing

### Feature Catalog

```rust
pub struct FeatureIndex {
    // Feature type registry
    feature_types: HashMap<FeatureType, FeatureTypeInfo>,
    
    // Feature instances
    features: HashMap<FeatureId, FeatureInstance>,
    
    // Temporal tracking
    feature_tracks: Vec<FeatureTrack>,
    
    // Spatial acceleration
    feature_rtree: RTree<FeatureInstance>,
}

pub struct FeatureInstance {
    feature_id: FeatureId,
    feature_type: FeatureType,
    
    // Spatiotemporal extent
    spatial_bounds: AABB,
    time_range: (f64, f64),
    
    // Properties
    properties: HashMap<String, Value>,
    
    // Data references
    chunk_refs: Vec<ChunkReference>,
}

pub struct FeatureTrack {
    track_id: TrackId,
    feature_type: FeatureType,
    
    // Track points
    trajectory: Vec<TrackPoint>,
    
    // Track properties
    birth_time: f64,
    death_time: Option<f64>,
    
    // Relationships
    parent_track: Option<TrackId>,
    child_tracks: Vec<TrackId>,
}
```

### Feature-Driven Access

```rust
impl FeatureIndex {
    pub fn find_features_by_type(&self, feature_type: FeatureType) -> Vec<&FeatureInstance> {
        self.features.values()
            .filter(|f| f.feature_type == feature_type)
            .collect()
    }
    
    pub fn find_features_in_region(&self, region: &AABB) -> Vec<&FeatureInstance> {
        self.feature_rtree.locate_in_envelope(&region.to_envelope())
            .collect()
    }
    
    pub fn get_feature_evolution(&self, feature_id: FeatureId) -> Option<Vec<ChunkReference>> {
        self.features.get(&feature_id)
            .map(|feature| {
                let track = self.find_track_for_feature(feature_id);
                self.collect_chunks_along_track(track)
            })
    }
}
```

## 5. S3-Optimized Storage Layout

### Object Key Design

```rust
pub struct S3StorageLayout {
    // Bucket configuration
    bucket_name: String,
    region: String,
    
    // Key prefixes
    index_prefix: String,      // "index/"
    data_prefix: String,       // "data/"
    metadata_prefix: String,   // "metadata/"
    
    // Partitioning strategy
    partitioning: PartitioningStrategy,
}

pub enum PartitioningStrategy {
    Temporal {
        // /data/year=2024/month=01/day=15/hour=14/
        granularity: TimeGranularity,
    },
    Spatial {
        // /data/x=10/y=20/z=30/
        grid_size: Vec3,
    },
    Hybrid {
        // /data/t=1000/region=northeast/level=2/
        time_buckets: u32,
        spatial_regions: u32,
    },
}

impl S3StorageLayout {
    pub fn build_chunk_key(&self, chunk: &ChunkMetadata) -> String {
        match &self.partitioning {
            PartitioningStrategy::Temporal { granularity } => {
                let time_path = self.time_to_path(chunk.time, granularity);
                format!("{}/{}/chunk_{}.dat", self.data_prefix, time_path, chunk.id)
            },
            PartitioningStrategy::Spatial { grid_size } => {
                let spatial_path = self.position_to_path(chunk.center, grid_size);
                format!("{}/{}/chunk_{}.dat", self.data_prefix, spatial_path, chunk.id)
            },
            PartitioningStrategy::Hybrid { time_buckets, spatial_regions } => {
                let time_bucket = (chunk.time / *time_buckets as f64) as u32;
                let region = self.compute_region(chunk.bounds, *spatial_regions);
                format!("{}/t={}/region={}/chunk_{}.dat", 
                    self.data_prefix, time_bucket, region, chunk.id)
            },
        }
    }
}
```

### Manifest Files

```rust
pub struct DataManifest {
    // Version and compatibility
    manifest_version: Version,
    created_at: DateTime<Utc>,
    
    // Chunk registry
    chunks: Vec<ChunkEntry>,
    
    // Object listing
    objects: HashMap<String, ObjectMetadata>,
    
    // Checksums for integrity
    chunk_checksums: HashMap<ChunkId, Checksum>,
}

pub struct ChunkEntry {
    chunk_id: ChunkId,
    
    // S3 location
    s3_key: String,
    byte_range: Option<(u64, u64)>, // For range requests
    
    // Chunk metadata
    compressed_size: u64,
    uncompressed_size: u64,
    compression: CompressionType,
    
    // Content description
    time_range: (f64, f64),
    spatial_bounds: AABB,
    data_types: Vec<DataType>,
}
```

### Multi-Part Upload Strategy

```rust
pub struct S3UploadStrategy {
    // Part size for multipart uploads
    part_size: u64, // e.g., 100MB
    
    // Concurrent upload limits
    max_concurrent_uploads: usize,
    
    pub async fn upload_simulation_data(&self, 
        simulation: &SimulationData,
        s3_client: &S3Client
    ) -> Result<UploadResult, S3Error> {
        // Build index first
        let index = self.build_index(simulation);
        
        // Upload in parallel
        let upload_tasks = simulation.chunks()
            .map(|chunk| self.upload_chunk(chunk, s3_client))
            .collect::<Vec<_>>();
        
        // Wait for all uploads
        let results = futures::future::join_all(upload_tasks).await;
        
        // Upload manifest and index
        self.upload_manifest(&index, s3_client).await?;
        
        Ok(UploadResult {
            total_objects: results.len(),
            total_bytes: results.iter().sum(),
        })
    }
}
```

## 6. Index Compression and Optimization

### Compressed Index Format

```rust
pub struct CompressedIndex {
    // Header (uncompressed)
    header: IndexHeader,
    
    // Compressed sections
    temporal_section: CompressedSection,
    spatial_section: CompressedSection,
    feature_section: CompressedSection,
    
    // Bloom filters for existence queries
    chunk_bloom: BloomFilter,
    feature_bloom: BloomFilter,
}

pub struct CompressedSection {
    compression_type: CompressionType,
    compressed_size: u64,
    uncompressed_size: u64,
    data: Vec<u8>,
}

impl CompressedIndex {
    pub fn compress(index: &SimulationIndex) -> Self {
        let temporal_data = bincode::serialize(&index.temporal_index).unwrap();
        let spatial_data = bincode::serialize(&index.spatial_index).unwrap();
        
        CompressedIndex {
            header: IndexHeader::from(index),
            temporal_section: Self::compress_section(temporal_data, CompressionType::Zstd),
            spatial_section: Self::compress_section(spatial_data, CompressionType::Lz4),
            feature_section: Self::compress_section(feature_data, CompressionType::Zstd),
            chunk_bloom: Self::build_bloom_filter(&index.chunks),
            feature_bloom: Self::build_bloom_filter(&index.features),
        }
    }
}
```

### Index Caching Strategy

```rust
pub struct IndexCache {
    // Memory cache
    memory_cache: LruCache<IndexKey, Arc<IndexSection>>,
    
    // Disk cache
    disk_cache: DiskCache,
    
    // Cache statistics
    stats: CacheStatistics,
    
    pub async fn get_index_section(&self, key: IndexKey) -> Result<Arc<IndexSection>, CacheError> {
        // Check memory cache
        if let Some(section) = self.memory_cache.get(&key) {
            self.stats.memory_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(section.clone());
        }
        
        // Check disk cache
        if let Some(section) = self.disk_cache.get(&key).await? {
            self.memory_cache.put(key.clone(), section.clone());
            self.stats.disk_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(section);
        }
        
        // Cache miss
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        Err(CacheError::Miss)
    }
}
```

## 7. Query Optimization

### Query Planning

```rust
pub struct QueryPlanner {
    // Index statistics
    index_stats: IndexStatistics,
    
    // Cost model
    cost_model: CostModel,
    
    pub fn plan_query(&self, query: &Query) -> QueryPlan {
        match query {
            Query::SpatialRange { bounds, time } => {
                self.plan_spatial_query(bounds, *time)
            },
            Query::TemporalRange { start, end, region } => {
                self.plan_temporal_query(*start, *end, region)
            },
            Query::FeatureBased { feature_type, constraints } => {
                self.plan_feature_query(feature_type, constraints)
            },
            Query::Composite { subqueries } => {
                self.plan_composite_query(subqueries)
            },
        }
    }
    
    fn estimate_cost(&self, plan: &QueryPlan) -> f64 {
        let io_cost = plan.estimated_chunks * self.cost_model.chunk_read_cost;
        let network_cost = plan.estimated_bytes * self.cost_model.byte_transfer_cost;
        let compute_cost = plan.estimated_operations * self.cost_model.operation_cost;
        
        io_cost + network_cost + compute_cost
    }
}
```

### Parallel Query Execution

```rust
pub struct ParallelQueryExecutor {
    // Thread pool for parallel execution
    thread_pool: ThreadPool,
    
    // S3 client pool
    s3_clients: Vec<S3Client>,
    
    pub async fn execute_query(&self, plan: QueryPlan) -> Result<QueryResult, QueryError> {
        // Decompose into parallel tasks
        let tasks = self.decompose_plan(plan);
        
        // Execute in parallel
        let results = tasks.into_par_iter()
            .map(|task| self.execute_task(task))
            .collect::<Result<Vec<_>, _>>()?;
        
        // Merge results
        self.merge_results(results)
    }
}
```

## 8. Index Maintenance

### Incremental Updates

```rust
pub struct IndexUpdater {
    // Current index
    current_index: Arc<RwLock<SimulationIndex>>,
    
    // Update log
    update_log: UpdateLog,
    
    pub async fn add_timestep(&self, timestep: SimulationTimestep) -> Result<(), UpdateError> {
        // Create chunks for new timestep
        let chunks = self.create_chunks(timestep);
        
        // Update indices
        let mut index = self.current_index.write().await;
        
        // Update temporal index
        index.temporal_index.add_timestep(timestep.time, &chunks);
        
        // Update spatial index
        for chunk in &chunks {
            index.spatial_index.insert(chunk);
        }
        
        // Extract and index features
        let features = self.extract_features(&timestep);
        for feature in features {
            index.feature_index.add_feature(feature);
        }
        
        // Log update
        self.update_log.record(UpdateRecord {
            timestamp: Utc::now(),
            operation: UpdateOp::AddTimestep(timestep.time),
            affected_chunks: chunks.len(),
        });
        
        Ok(())
    }
}
```

### Index Optimization

```rust
pub struct IndexOptimizer {
    pub fn optimize_index(&self, index: &mut SimulationIndex) -> OptimizationResult {
        let mut result = OptimizationResult::default();
        
        // Rebalance spatial tree
        if self.should_rebalance_spatial(&index.spatial_index) {
            self.rebalance_octree(&mut index.spatial_index);
            result.spatial_rebalanced = true;
        }
        
        // Compact temporal index
        if self.should_compact_temporal(&index.temporal_index) {
            self.compact_temporal_index(&mut index.temporal_index);
            result.temporal_compacted = true;
        }
        
        // Rebuild bloom filters
        self.rebuild_bloom_filters(index);
        
        // Update statistics
        index.metadata.last_optimized = Utc::now();
        
        result
    }
}
```

## 9. Best Practices

1. **Design for Cloud**: Optimize for S3's strengths (parallel reads, range requests)
2. **Hierarchical Organization**: Use multiple index levels for different access patterns
3. **Compression is Key**: Compress indices aggressively but allow partial decompression
4. **Cache Strategically**: Cache frequently accessed index sections
5. **Plan for Growth**: Design indices that can handle dataset growth
6. **Version Everything**: Include version info to handle schema evolution
7. **Minimize Metadata Overhead**: Keep per-chunk metadata compact

## 10. Example Implementation

```rust
// Example: Building and using a simulation index
pub async fn index_simulation_example() -> Result<(), Box<dyn Error>> {
    // Create index builder
    let builder = IndexBuilder::new()
        .with_temporal_resolution(0.1)
        .with_spatial_grid_size(100.0)
        .with_compression(CompressionType::Zstd);
    
    // Build index from simulation data
    let simulation = load_simulation_data()?;
    let index = builder.build(&simulation)?;
    
    // Upload to S3
    let s3_client = S3Client::new(Region::UsEast1);
    let storage = S3StorageLayout::new("my-simulation-bucket")
        .with_partitioning(PartitioningStrategy::Hybrid {
            time_buckets: 100,
            spatial_regions: 8,
        });
    
    storage.upload_index(&index, &s3_client).await?;
    
    // Query example
    let query_engine = QueryEngine::new(index, s3_client);
    
    let results = query_engine.query()
        .spatial_range(AABB::new(Vec3::ZERO, Vec3::new(100.0, 100.0, 100.0)))
        .time_range(0.0, 10.0)
        .with_features(vec![FeatureType::Vortex])
        .execute()
        .await?;
    
    println!("Found {} chunks matching query", results.chunks.len());
    
    Ok(())
}
```

## References

1. "Designing Data-Intensive Applications" - Martin Kleppmann
2. "Database Internals" - Alex Petrov
3. "AWS S3 Best Practices" - AWS Documentation
4. "Spatial Indexing Techniques" - Hanan Samet
5. "Time Series Database Internals" - InfluxData
6. "Apache Parquet Design" - Apache Parquet Documentation