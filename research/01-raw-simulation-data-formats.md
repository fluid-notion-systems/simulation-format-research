# Raw Simulation Data Formats

## Overview

Raw simulation data represents the fundamental physical quantities computed during numerical simulations. This document provides a comprehensive analysis of data types, formats, and storage strategies for various simulation scenarios, with a primary focus on fluid dynamics.

## Core Physical Quantities

### 1. Velocity Fields

**Description**: Vector fields representing fluid motion at each point in space.

**Data Structure**:
```
struct VelocityField {
    u: f32/f64,  // x-component
    v: f32/f64,  // y-component
    w: f32/f64,  // z-component (3D only)
}
```

**Storage Considerations**:
- Typically 12-24 bytes per grid point (3D, depending on precision)
- Can be stored as Structure of Arrays (SoA) or Array of Structures (AoS)
- SoA often better for GPU processing and compression
- Staggered grid storage (MAC grid) vs collocated storage

**Compression Strategies**:
- Velocity gradients often smooth → good wavelet compression
- Predictive coding between timesteps
- Quantization with acceptable error bounds

### 2. Pressure Fields

**Description**: Scalar field representing fluid pressure at each point.

**Data Structure**:
```
struct PressureField {
    p: f32/f64,  // pressure value
}
```

**Storage Considerations**:
- 4-8 bytes per grid point
- Often stored at cell centers
- Relative pressure vs absolute pressure
- Gauge pressure considerations

**Special Requirements**:
- High precision needed for incompressible flow
- Boundary condition storage
- Pressure correction fields for iterative solvers

### 3. Density Fields

**Description**: Scalar field for fluid density, critical for compressible flows.

**Data Structure**:
```
struct DensityField {
    rho: f32/f64,  // density value
}
```

**Applications**:
- Variable density flows
- Multiphase simulations
- Buoyancy-driven flows

### 4. Temperature Fields

**Description**: Scalar field for thermal simulations.

**Data Structure**:
```
struct TemperatureField {
    T: f32/f64,  // temperature value
}
```

**Coupling Considerations**:
- Coupled with velocity through buoyancy
- Affects material properties
- Energy equation coupling

### 5. Vorticity Fields

**Description**: Derived quantity representing local rotation.

**Calculation**:
```
ω = ∇ × v
```

**Storage Options**:
- Store computed vorticity (redundant but fast)
- Compute on-demand from velocity
- Store only high-vorticity regions

### 6. Turbulence Quantities

**For RANS/LES simulations**:
- Turbulent kinetic energy (k)
- Dissipation rate (ε)
- Specific dissipation (ω)
- Reynolds stresses (6 components)
- Subgrid-scale stresses

## Grid Types and Storage Patterns

### 1. Structured Grids

**Regular Cartesian**:
```
data[i][j][k] = value at (i*dx, j*dy, k*dz)
```

**Advantages**:
- Simple indexing
- Cache-friendly access patterns
- Efficient compression
- Easy parallelization

**Storage Format**:
```rust
struct CartesianGrid3D {
    nx: usize, ny: usize, nz: usize,
    dx: f64, dy: f64, dz: f64,
    origin: Vec3,
    data: Vec<T>,  // Flattened array
}
```

### 2. Unstructured Grids

**Storage Components**:
- Vertex coordinates
- Cell connectivity
- Face information
- Cell-to-cell adjacency

**Format Example**:
```rust
struct UnstructuredGrid {
    vertices: Vec<Vec3>,
    cells: Vec<CellConnectivity>,
    faces: Vec<Face>,
    cell_data: HashMap<String, Vec<f64>>,
    vertex_data: HashMap<String, Vec<f64>>,
}
```

### 3. Adaptive Mesh Refinement (AMR)

**Hierarchical Structure**:
- Octree/Quadtree based
- Multiple resolution levels
- Parent-child relationships

**Storage Challenges**:
- Efficient tree traversal
- Ghost cell management
- Load balancing

## Time-Series Considerations

### Temporal Storage Strategies

1. **Full Snapshots**:
   - Complete state at each timestep
   - High storage, simple access
   - Good for restart capability

2. **Delta Compression**:
   - Store base state + changes
   - Reduced storage
   - More complex reconstruction

3. **Keyframe + Interpolation**:
   - Store key states
   - Interpolate intermediate states
   - Balance between storage and accuracy

### Temporal Access Patterns

```rust
struct TemporalDataset {
    timesteps: Vec<f64>,
    snapshots: Vec<SimulationState>,
    metadata: TimeSeriesMetadata,
}

struct TimeSeriesMetadata {
    dt: f64,
    total_time: f64,
    snapshot_interval: usize,
    compression_method: CompressionType,
}
```

## Data Precision Requirements

### Single vs Double Precision

**Single Precision (f32)**:
- 4 bytes per value
- ~7 decimal digits
- Sufficient for visualization
- Faster GPU processing

**Double Precision (f64)**:
- 8 bytes per value
- ~15 decimal digits
- Required for:
  - Long time integrations
  - Small-scale phenomena
  - Iterative solver convergence

### Mixed Precision Strategies

```rust
struct MixedPrecisionField {
    high_precision_regions: Vec<Region>,
    low_precision_data: Vec<f32>,
    high_precision_data: HashMap<RegionId, Vec<f64>>,
}
```

## Compression Techniques

### 1. Lossless Compression

**Methods**:
- DEFLATE/zlib
- LZ4 (fast decompression)
- Zstandard (good ratio/speed balance)

**Use Cases**:
- Initial conditions
- Boundary conditions
- Critical validation data

### 2. Lossy Compression

**Methods**:
- Wavelet compression
- DCT-based compression
- Quantization + entropy coding

**Error Control**:
```rust
struct CompressionConfig {
    absolute_tolerance: f64,
    relative_tolerance: f64,
    preserve_features: bool,
    feature_threshold: f64,
}
```

### 3. Domain-Specific Compression

**Techniques**:
- Exploit incompressibility (∇·v = 0)
- Symmetry exploitation
- Periodic boundary compression
- Statistical redundancy removal

## Storage Format Recommendations

### 1. HDF5 Format

**Structure**:
```
/simulation
  /metadata
    - grid_type
    - dimensions
    - physical_parameters
  /time_000000
    /velocity
      - u
      - v
      - w
    /pressure
    /density
  /time_000001
    ...
```

**Advantages**:
- Self-describing
- Portable
- Parallel I/O support
- Built-in compression

### 2. Custom Binary Format

**Header Structure**:
```c
struct SimulationHeader {
    magic_number: u32,
    version: u16,
    grid_type: u8,
    precision: u8,
    dimensions: [u32; 3],
    num_variables: u16,
    num_timesteps: u32,
    compression: u8,
}
```

### 3. Cloud-Optimized Formats

**Zarr Format**:
- Chunked storage
- Cloud-native
- Parallel access
- Multiple compression codecs

**Parquet for Time Series**:
- Columnar storage
- Efficient compression
- Query optimization
- Streaming support

## Performance Considerations

### Memory Layout Optimization

```rust
// AoS - Better for random access
struct PointData {
    velocity: Vec3,
    pressure: f64,
    density: f64,
}

// SoA - Better for SIMD and GPU
struct FieldData {
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    velocity_z: Vec<f64>,
    pressure: Vec<f64>,
    density: Vec<f64>,
}
```

### Access Pattern Optimization

1. **Spatial Locality**:
   - Morton/Z-order curves
   - Hilbert curves
   - Cache-oblivious layouts

2. **Temporal Locality**:
   - Group nearby timesteps
   - Predictive prefetching
   - Sliding window buffers

## Integration with Simulation Codes

### Common Simulation Output Formats

1. **OpenFOAM Format**:
   - ASCII/Binary options
   - Time directories
   - Field files

2. **VTK/VTU Format**:
   - XML-based
   - Unstructured grid support
   - ParaView compatible

3. **CGNS Format**:
   - CFD General Notation System
   - Hierarchical structure
   - Standardized naming

### Data Pipeline Example

```rust
trait SimulationDataReader {
    fn read_timestep(&self, time: f64) -> Result<SimulationState>;
    fn read_variable(&self, name: &str, time: f64) -> Result<Field>;
    fn get_metadata(&self) -> &SimulationMetadata;
}

struct SimulationState {
    time: f64,
    fields: HashMap<String, Field>,
    grid: Grid,
}
```

## Future Considerations

### Emerging Requirements

1. **Machine Learning Integration**:
   - Feature extraction storage
   - Training data organization
   - Reduced-order model coefficients

2. **Real-time Visualization**:
   - Progressive streaming formats
   - Adaptive resolution
   - GPU-direct storage

3. **Uncertainty Quantification**:
   - Ensemble storage
   - Statistical moment fields
   - Probability distributions

### Scalability Challenges

1. **Exascale Simulations**:
   - Petabyte datasets
   - Distributed storage
   - In-situ processing

2. **Multi-physics Coupling**:
   - Heterogeneous data types
   - Different time scales
   - Interface tracking

## Best Practices Summary

1. **Choose appropriate precision** based on physics and accuracy requirements
2. **Use hierarchical storage** for multi-resolution access
3. **Implement compression** with controlled error bounds
4. **Design for parallel I/O** from the start
5. **Include comprehensive metadata** for reproducibility
6. **Plan for temporal access patterns** in storage layout
7. **Consider cloud storage** early in design
8. **Validate compression errors** against physical constraints

## References and Further Reading

1. "The HDF5 Library & File Format" - The HDF Group
2. "Compression of Scientific Data" - Lindstrom & Isenburg
3. "OpenVDB: An Open-Source Data Structure and Toolkit" - Museth
4. "Cloud-Optimized GeoTIFF" - Even Rouault
5. "The Zarr Specification" - Zarr Development Team
6. "Parallel I/O for High Performance Computing" - Gropp et al.