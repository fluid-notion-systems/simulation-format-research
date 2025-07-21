# Simulation Data Format/Data Lake

A comprehensive system for ingestion of massive datasets from simulations, with initial focus on fluid simulation and extensibility to fire, smoke, gas, and thermodynamic simulations.

## Overview

This project aims to create a modern, scalable data format and storage system for large-scale scientific simulations, enabling efficient storage, streaming, and visualization of time-varying volumetric data.

## Core Features

### 1. Raw Simulation Data Support

Support for various simulation types and their associated data:

- **Currently Implemented**: Lattice Boltzmann Method (LBM)
- **Planned Support**: 
  - Smoothed Particle Hydrodynamics (SPH)
  - Material Point Method (MPM)
  - Finite Element Method (FEM)
  - Fire, smoke, gas, and thermodynamic simulations

**Data Types**:
- Velocity fields
- Pressure fields
- Temperature data
- Density information
- Other physics-specific quantities

### 2. Condensed Information Formats

Derived data for efficient visualization:

- **Isosurfaces**: Extracted surface representations
- **Vorticity**: Rotational flow features
- **Feature extraction**: Common visualization primitives

**Progressive Refinement**:
- Index-based format for quick initial display
- Nanite/meshlet-like functionality for successive refinement
- Integration with Bevy's meshlet implementation

### 3. Cloud-Native Storage Design

**S3-Compatible Architecture**:
- Download index data from S3
- Index files contain pointers to different datasets
- Optimized for cloud storage patterns

**Temporal Considerations**:
- Support for simulation "moments" in time
- Linear temporal progression
- Efficient time-based access patterns

### 4. Spatial-Aware Streaming (Advanced)

Future capability for intelligent spatial data streaming:

- Similar to Google Maps tiling or Cesium terrain streaming
- Full 3D spatial awareness
- Stream only visible/critical portions based on viewer position
- Progressive level-of-detail loading

## Technical Implementation

### Architectural Choices for Data Ingestion

Cross-language compatible design using:
- **Channels/Crossbeam**: For Rust-based parallel processing
- **Parquet**: Columnar storage format
- **Time-series optimizations**: For temporal data
- **Avro**: Schema evolution support (under consideration)

### GPU-Driven Data Processing

- Mesh simplification on GPU
- Hierarchical data structure generation
- Real-time optimization for visualization
- Efficient processing pipelines

### Technology Research

**Open Source Simulation Software**:
- Genesis
- OpenFOAM
- Other major simulation packages

**Data Structure Technologies**:
- OpenVDB architecture and techniques
- Multi-resolution data structures from Taichi
- Genesis implementation patterns

## Memory Format Optimization

### Current Storage Analysis

Current simulations use mixed precision:
- `fp16` (half precision)
- `fp32` (single precision)  
- `fp64` (double precision)

### Identified Inefficiencies

Significant wasted bits in storing:

1. **Rotations**: 
   - Currently stored as normalized quaternions (4 floats)
   - Only 3 degrees of freedom needed
   - Potential for index-based encoding

2. **Velocity Vectors**:
   - Fluid simulations have bounded velocities (e.g., max 2.0 m/s)
   - Full float precision unnecessary
   - Domain-specific compression possible

### Proposed Optimization: Fibonacci Sphere Encoding

**Requirements**:
- Store rotations/normalized vectors as indices
- Two-way mapping between index and rotation
- Efficient encoding/decoding
- Controllable precision

**Benefits**:
- Significant memory reduction
- Cache-friendly access patterns
- GPU-optimized lookups

## Research Documentation

For detailed technical deep dives into all aspects of this system, see the [Research Index](research/index.md).