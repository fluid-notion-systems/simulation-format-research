# Architectural Choices

## Overview

This document explores architectural decisions for building a comprehensive simulation data ingestion and processing system. It focuses on concurrency models, data formats, cross-language compatibility, and scalable system design patterns suitable for handling massive simulation datasets.

## 1. Data Ingestion Architecture

### Pipeline Design Patterns

#### Stream Processing Architecture
```rust
pub struct StreamingIngestionPipeline {
    // Input stage - receives raw simulation data
    input_receiver: Receiver<RawSimulationData>,
    
    // Processing stages
    parser: DataParser,
    validator: DataValidator,
    transformer: DataTransformer,
    compressor: DataCompressor,
    
    // Output stage - writes to storage
    writer: StorageWriter,
    
    // Backpressure management
    buffer_manager: BufferManager,
}

impl StreamingIngestionPipeline {
    pub async fn process(&mut self) {
        while let Ok(raw_data) = self.input_receiver.recv() {
            // Apply backpressure if needed
            self.buffer_manager.wait_if_full().await;
            
            // Pipeline stages
            let parsed = self.parser.parse(raw_data)?;
            let validated = self.validator.validate(parsed)?;
            let transformed = self.transformer.transform(validated)?;
            let compressed = self.compressor.compress(transformed)?;
            
            // Write to storage
            self.writer.write(compressed).await?;
        }
    }
}
```

#### Batch Processing Architecture
```rust
pub struct BatchIngestionSystem {
    // Batch accumulator
    batch_builder: BatchBuilder,
    batch_size: usize,
    batch_timeout: Duration,
    
    // Processing engine
    processing_engine: BatchProcessor,
    
    // Coordination
    scheduler: BatchScheduler,
}

impl BatchIngestionSystem {
    pub async fn ingest(&mut self, data: SimulationData) {
        self.batch_builder.add(data);
        
        if self.batch_builder.is_full() || self.batch_builder.is_timeout() {
            let batch = self.batch_builder.build();
            self.scheduler.schedule(batch);
        }
    }
}
```

### Hybrid Architecture
```rust
pub struct HybridIngestionArchitecture {
    // Real-time path for critical data
    realtime_pipeline: StreamingIngestionPipeline,
    
    // Batch path for bulk data
    batch_system: BatchIngestionSystem,
    
    // Router to decide path
    router: DataRouter,
    
    pub async fn ingest(&mut self, data: IncomingData) {
        match self.router.route(&data) {
            RoutingDecision::Realtime => {
                self.realtime_pipeline.process(data).await
            },
            RoutingDecision::Batch => {
                self.batch_system.ingest(data).await
            },
        }
    }
}
```

## 2. Concurrency Models

### Channels-Based Architecture

#### Rust std::sync::mpsc
```rust
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;

pub struct ChannelBasedProcessor {
    // Multiple producers, single consumer
    sender: Sender<SimulationFrame>,
    receiver: Receiver<SimulationFrame>,
}

impl ChannelBasedProcessor {
    pub fn spawn_pipeline(num_workers: usize) -> Self {
        let (tx, rx) = channel();
        
        // Spawn worker threads
        for i in 0..num_workers {
            let tx_clone = tx.clone();
            thread::spawn(move || {
                loop {
                    if let Some(frame) = generate_simulation_frame() {
                        tx_clone.send(frame).unwrap();
                    }
                }
            });
        }
        
        ChannelBasedProcessor {
            sender: tx,
            receiver: rx,
        }
    }
}
```

#### Crossbeam Channels
```rust
use crossbeam::channel::{bounded, unbounded, select};

pub struct CrossbeamPipeline {
    // Bounded channel for backpressure
    data_channel: (Sender<DataChunk>, Receiver<DataChunk>),
    
    // Unbounded channel for control messages
    control_channel: (Sender<ControlMsg>, Receiver<ControlMsg>),
}

impl CrossbeamPipeline {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            data_channel: bounded(buffer_size),
            control_channel: unbounded(),
        }
    }
    
    pub fn process(&self) {
        loop {
            select! {
                recv(self.data_channel.1) -> msg => {
                    match msg {
                        Ok(chunk) => self.process_chunk(chunk),
                        Err(_) => break,
                    }
                },
                recv(self.control_channel.1) -> msg => {
                    match msg {
                        Ok(ControlMsg::Shutdown) => break,
                        Ok(ControlMsg::Flush) => self.flush(),
                        _ => continue,
                    }
                },
            }
        }
    }
}
```

### Actor Model with Actix
```rust
use actix::prelude::*;

pub struct DataIngestionActor {
    buffer: Vec<SimulationData>,
    storage: Addr<StorageActor>,
}

impl Actor for DataIngestionActor {
    type Context = Context<Self>;
}

impl Handler<IngestData> for DataIngestionActor {
    type Result = ();
    
    fn handle(&mut self, msg: IngestData, ctx: &mut Context<Self>) {
        self.buffer.push(msg.data);
        
        if self.buffer.len() >= BATCH_SIZE {
            let batch = std::mem::take(&mut self.buffer);
            self.storage.do_send(StoreBatch { data: batch });
        }
    }
}
```

### Async/Await with Tokio
```rust
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

pub struct AsyncIngestionSystem {
    // Async channels
    data_tx: mpsc::Sender<SimulationData>,
    data_rx: mpsc::Receiver<SimulationData>,
    
    // Task management
    tasks: JoinSet<Result<(), ProcessingError>>,
}

impl AsyncIngestionSystem {
    pub async fn run(&mut self) {
        // Spawn processing tasks
        for i in 0..num_cpus::get() {
            let mut rx = self.data_rx.clone();
            
            self.tasks.spawn(async move {
                while let Some(data) = rx.recv().await {
                    process_data_async(data).await?;
                }
                Ok(())
            });
        }
        
        // Wait for all tasks
        while let Some(result) = self.tasks.join_next().await {
            result??;
        }
    }
}
```

## 3. Data Format Selection

### Apache Parquet

#### Advantages
- **Columnar Storage**: Excellent compression for similar data
- **Schema Evolution**: Add/remove columns without rewriting
- **Predicate Pushdown**: Skip irrelevant data during reads
- **Wide Ecosystem**: Spark, Pandas, Arrow support

#### Implementation
```rust
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

pub struct ParquetWriter {
    writer: ArrowWriter<File>,
    properties: WriterProperties,
}

impl ParquetWriter {
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        
        let properties = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD)
            .set_dictionary_enabled(true)
            .set_statistics_enabled(true)
            .build();
        
        let writer = ArrowWriter::try_new(file, schema, Some(properties))?;
        
        Ok(Self { writer, properties })
    }
    
    pub fn write_batch(&mut self, batch: RecordBatch) -> Result<()> {
        self.writer.write(&batch)?;
        Ok(())
    }
}
```

### Apache Avro

#### Advantages
- **Schema Evolution**: Rich schema evolution rules
- **Compact Binary**: Smaller than JSON
- **Self-Describing**: Schema stored with data
- **RPC Support**: Can be used for services

#### Implementation
```rust
use apache_avro::{Schema, Writer, types::Record};

pub struct AvroIngestion {
    schema: Schema,
    writer: Writer<File>,
}

impl AvroIngestion {
    pub fn write_simulation_frame(&mut self, frame: SimulationFrame) -> Result<()> {
        let mut record = Record::new(&self.schema).unwrap();
        
        record.put("timestamp", frame.timestamp);
        record.put("velocity_field", frame.velocity_to_avro());
        record.put("pressure_field", frame.pressure_to_avro());
        
        self.writer.append(record)?;
        Ok(())
    }
}
```

### Format Comparison for Simulation Data

| Feature | Parquet | Avro | HDF5 | Custom Binary |
|---------|---------|------|------|---------------|
| Compression | Excellent | Good | Good | Variable |
| Schema Evolution | Yes | Yes | Limited | Manual |
| Cross-Language | Yes | Yes | Yes | Depends |
| Streaming | Limited | Yes | No | Yes |
| Random Access | Column-level | No | Yes | Yes |
| Cloud-Native | Yes | Yes | No | Depends |

## 4. Time Series Architecture

### Time-Partitioned Storage
```rust
pub struct TimePartitionedStorage {
    // Partition strategy
    partition_duration: Duration,
    
    // Active partitions
    active_partitions: HashMap<PartitionKey, Partition>,
    
    // Closed partitions
    archived_partitions: BTreeMap<TimeRange, ArchivedPartition>,
}

impl TimePartitionedStorage {
    pub fn write(&mut self, time: f64, data: SimulationData) -> Result<()> {
        let partition_key = self.compute_partition_key(time);
        
        let partition = self.active_partitions
            .entry(partition_key)
            .or_insert_with(|| Partition::new(partition_key));
        
        partition.append(time, data)?;
        
        // Check if partition should be closed
        if partition.should_close() {
            self.close_partition(partition_key)?;
        }
        
        Ok(())
    }
}
```

### Time Series Index
```rust
pub struct TimeSeriesIndex {
    // B-tree for time-based lookup
    time_index: BTreeMap<OrderedFloat<f64>, DataLocation>,
    
    // Bloom filters for existence queries
    bloom_filters: Vec<BloomFilter>,
    
    // Statistics for query optimization
    statistics: TimeSeriesStats,
}

pub struct TimeSeriesStats {
    min_time: f64,
    max_time: f64,
    total_points: u64,
    avg_time_delta: f64,
    time_distribution: Histogram,
}
```

## 5. Cross-Language Compatibility

### Language-Agnostic Protocols

#### gRPC Service Definition
```protobuf
syntax = "proto3";

service SimulationIngestion {
    rpc StreamData(stream SimulationFrame) returns (IngestionResponse);
    rpc BatchIngest(BatchRequest) returns (BatchResponse);
    rpc GetStatus(StatusRequest) returns (StatusResponse);
}

message SimulationFrame {
    double timestamp = 1;
    bytes velocity_data = 2;  // Compressed array
    bytes pressure_data = 3;  // Compressed array
    Metadata metadata = 4;
}
```

#### REST API Design
```rust
use warp::{Filter, Reply};

pub fn create_ingestion_api() -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
    let ingest_route = warp::post()
        .and(warp::path("ingest"))
        .and(warp::body::json())
        .and_then(handle_ingest);
    
    let status_route = warp::get()
        .and(warp::path("status"))
        .and_then(handle_status);
    
    ingest_route.or(status_route)
}
```

### Language Bindings

#### Python Integration
```python
# Using PyO3 for Rust-Python binding
import simulation_ingestor

# Initialize the ingestor
ingestor = simulation_ingestor.Ingestor(
    config_path="config.yaml",
    num_workers=4
)

# Ingest data
for frame in simulation_frames:
    ingestor.ingest(
        timestamp=frame.time,
        velocity=frame.velocity,
        pressure=frame.pressure
    )

# Get statistics
stats = ingestor.get_stats()
```

#### C++ Integration
```cpp
// Using C ABI for compatibility
extern "C" {
    SimulationIngestor* create_ingestor(const char* config_path);
    void ingest_frame(SimulationIngestor* ingestor, const SimFrame* frame);
    void destroy_ingestor(SimulationIngestor* ingestor);
}

class IngestorWrapper {
    SimulationIngestor* ingestor;
    
public:
    IngestorWrapper(const std::string& config) {
        ingestor = create_ingestor(config.c_str());
    }
    
    void ingest(const SimulationFrame& frame) {
        SimFrame c_frame = frame.to_c_struct();
        ingest_frame(ingestor, &c_frame);
    }
    
    ~IngestorWrapper() {
        destroy_ingestor(ingestor);
    }
};
```

## 6. System Architecture Patterns

### Microservices Architecture
```rust
pub struct MicroservicesArchitecture {
    // Service mesh
    ingestion_service: ServiceEndpoint,
    processing_service: ServiceEndpoint,
    storage_service: ServiceEndpoint,
    query_service: ServiceEndpoint,
    
    // Service discovery
    registry: ServiceRegistry,
    
    // Communication
    message_bus: MessageBus,
}

impl MicroservicesArchitecture {
    pub async fn process_data(&self, data: SimulationData) -> Result<()> {
        // Ingestion service
        let ingested = self.ingestion_service
            .call(IngestRequest { data })
            .await?;
        
        // Processing service (async)
        self.message_bus
            .publish(ProcessingRequest { data: ingested })
            .await?;
        
        Ok(())
    }
}
```

### Lambda Architecture
```rust
pub struct LambdaArchitecture {
    // Batch layer
    batch_pipeline: BatchPipeline,
    batch_storage: BatchStorage,
    
    // Speed layer
    stream_processor: StreamProcessor,
    realtime_storage: RealtimeStorage,
    
    // Serving layer
    query_merger: QueryMerger,
}

impl LambdaArchitecture {
    pub async fn query(&self, request: QueryRequest) -> QueryResult {
        // Query both layers
        let batch_result = self.batch_storage.query(&request).await?;
        let realtime_result = self.realtime_storage.query(&request).await?;
        
        // Merge results
        self.query_merger.merge(batch_result, realtime_result)
    }
}
```

## 7. Performance Considerations

### Memory Management
```rust
pub struct MemoryOptimizedIngestion {
    // Memory pools
    buffer_pool: BufferPool,
    
    // Zero-copy where possible
    use_mmap: bool,
    
    // Memory limits
    max_memory: usize,
    current_usage: AtomicUsize,
}

impl MemoryOptimizedIngestion {
    pub fn allocate_buffer(&self, size: usize) -> Result<PooledBuffer> {
        let current = self.current_usage.load(Ordering::Relaxed);
        
        if current + size > self.max_memory {
            return Err(IngestError::MemoryLimit);
        }
        
        self.buffer_pool.allocate(size)
    }
}
```

### CPU Optimization
```rust
use rayon::prelude::*;

pub struct ParallelProcessor {
    thread_pool: ThreadPool,
    
    pub fn process_batch(&self, batch: Vec<SimulationFrame>) -> Result<Vec<ProcessedFrame>> {
        batch.par_iter()
            .map(|frame| self.process_frame(frame))
            .collect()
    }
    
    fn process_frame(&self, frame: &SimulationFrame) -> Result<ProcessedFrame> {
        // Use SIMD where available
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return self.process_frame_avx2(frame);
            }
        }
        
        self.process_frame_scalar(frame)
    }
}
```

## 8. Scalability Patterns

### Horizontal Scaling
```rust
pub struct HorizontalScalingStrategy {
    // Consistent hashing for distribution
    hash_ring: ConsistentHashRing,
    
    // Node management
    nodes: Vec<NodeInfo>,
    
    pub fn route_data(&self, data: &SimulationData) -> NodeId {
        let key = self.compute_routing_key(data);
        self.hash_ring.get_node(key)
    }
    
    pub fn add_node(&mut self, node: NodeInfo) {
        self.nodes.push(node.clone());
        self.hash_ring.add_node(node.id);
        self.rebalance();
    }
}
```

### Vertical Scaling
```rust
pub struct VerticalScalingOptimizations {
    // NUMA awareness
    numa_nodes: Vec<NumaNode>,
    
    // CPU affinity
    cpu_affinity: CpuSet,
    
    pub fn optimize_for_hardware(&mut self) {
        // Detect NUMA topology
        self.numa_nodes = detect_numa_topology();
        
        // Pin threads to cores
        for (thread_id, core_id) in self.compute_thread_affinity() {
            set_thread_affinity(thread_id, core_id);
        }
        
        // Optimize memory allocation
        self.configure_numa_allocation();
    }
}
```

## 9. Fault Tolerance

### Checkpointing Strategy
```rust
pub struct CheckpointManager {
    checkpoint_interval: Duration,
    checkpoint_storage: CheckpointStorage,
    
    pub async fn checkpoint(&self, state: &IngestionState) -> Result<CheckpointId> {
        let checkpoint = Checkpoint {
            timestamp: Utc::now(),
            state: state.clone(),
            metadata: self.collect_metadata(),
        };
        
        self.checkpoint_storage.store(checkpoint).await
    }
    
    pub async fn recover(&self) -> Result<IngestionState> {
        let latest = self.checkpoint_storage.get_latest().await?;
        
        // Replay from checkpoint
        self.replay_from(latest.state, latest.timestamp).await
    }
}
```

## 10. Best Practices

### Architecture Decision Records (ADR)

```markdown
# ADR-001: Use Parquet for Long-term Storage

## Status
Accepted

## Context
Need efficient storage format for simulation time series data.

## Decision
Use Apache Parquet with Zstandard compression.

## Consequences
- Excellent compression ratios
- Good query performance
- Limited streaming support
- Need batch accumulation
```

### Design Principles

1. **Separation of Concerns**
   - Ingestion, processing, storage as separate services
   - Clear interfaces between components

2. **Fail-Fast Principle**
   - Validate early in pipeline
   - Clear error propagation

3. **Observability First**
   - Metrics at every stage
   - Distributed tracing
   - Structured logging

4. **Schema Evolution**
   - Forward/backward compatibility
   - Versioned schemas
   - Migration strategies

## References

1. "Designing Data-Intensive Applications" - Martin Kleppmann
2. "The Architecture of Open Source Applications" - Various
3. "Building Microservices" - Sam Newman
4. "Stream Processing with Apache Flink" - Hueske & Kalavri
5. "High Performance Spark" - Karau & Warren