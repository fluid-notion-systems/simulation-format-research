# Simulation Metrics for Big Iron GPU Configurations

## Executive Summary

This document estimates the computational capabilities and domain coverage for fluid simulations using various GPU configurations. We analyze single and multi-GPU setups using NVIDIA A200/B200 and AMD MI300X GPUs, targeting resolution granularities from 1mm to 10cm for ocean-scale simulations.

## Key Assumptions

### Memory Requirements
- **Conservative**: 200 bytes per lattice node (current implementations)
- **Optimized**: 116 bits (14.5 bytes) per node (research target with compression) (Nick: look into FLuidX3d implementation, and Dr Moritz Lemanns research  into memory optimizations, esoteric pull (what is that exactly..))
- **Production**: 50 bytes per node (realistic near-term with basic optimizations)

### Performance Baselines
- Single GPU target: 1000+ MLUPS (Million Lattice Updates Per Second)
- Multi-GPU scaling efficiency: 85-90%
- Memory bandwidth utilization: 70-80%

## GPU Specifications

### NVIDIA GPUs
- **A200 (40GB)**: 40GB HBM2e, 1.56 TB/s bandwidth
- **A100 (80GB)**: 80GB HBM2e, 2.04 TB/s bandwidth
- **H100 (80GB)**: 80GB HBM3, 3.35 TB/s bandwidth
- **B200 (192GB)**: 192GB HBM3e, 8 TB/s bandwidth (Blackwell)

### AMD GPUs
- **MI250X**: 128GB HBM2e, 3.28 TB/s bandwidth
- **MI300X**: 192GB HBM3, 5.3 TB/s bandwidth

### Cerebras Wafer-Scale Engine
- **WSE-3**: 44GB on-chip SRAM, 21 PB/s memory bandwidth, 900,000 AI cores
- **Die Size**: 46,225 mm² (57x larger than H100)
- **Transistors**: 4 trillion (50x more than H100)
- **External Memory**: Up to 1.2 PB via MemoryX units

## Single GPU Configurations

### 1x A200 (40GB)
**Memory-Limited Domain Sizes:**
- @ 200 bytes/node: 200M nodes
- @ 50 bytes/node: 800M nodes
- @ 14.5 bytes/node: 2.76B nodes

**Physical Domains (50 bytes/node):**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 10cm | 200m × 200m × 20m | Coarse surf dynamics |
| 5cm | 120m × 120m × 15m | Basic wave resolution |
| 2cm | 60m × 60m × 10m | Detailed waves |
| 1cm | 40m × 40m × 5m | High-detail breaking |
| 5mm | 25m × 25m × 3m | Research quality |

### 1x B200 (192GB)
**Memory-Limited Domain Sizes:**
- @ 200 bytes/node: 960M nodes
- @ 50 bytes/node: 3.84B nodes
- @ 14.5 bytes/node: 13.2B nodes

**Physical Domains (50 bytes/node):**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 10cm | 340m × 340m × 35m | Large reef section |
| 5cm | 220m × 220m × 25m | Full surf break |
| 2cm | 120m × 120m × 15m | Detailed surf break |
| 1cm | 80m × 80m × 10m | Ultra-detailed |
| 5mm | 50m × 50m × 6m | Maximum detail |

### 1x MI300X (192GB)
Similar to B200 in memory capacity, slightly lower bandwidth:
- Performance: ~90% of B200 due to bandwidth difference
- Same domain sizes as B200
- Better price/performance ratio

## Multi-GPU Configurations

### 2x GPU Systems

**2x A200 (80GB total)**
- Domain: 2x single GPU with 90% efficiency
- Communication overhead for boundaries
- Best for long, thin domains

**2x B200 (384GB total)**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 5cm | 300m × 300m × 35m | Regional scale |
| 2cm | 170m × 170m × 20m | High-detail region |
| 1cm | 110m × 110m × 14m | Research grade |

### 4x GPU Systems

**4x A200 (160GB total)**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 5cm | 200m × 200m × 25m | Multi-break system |
| 2cm | 100m × 100m × 15m | Detailed reef |
| 1cm | 65m × 65m × 8m | High-res section |

**4x B200 (768GB total)**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 5cm | 400m × 400m × 50m | Large reef complex |
| 2cm | 240m × 240m × 30m | Detailed complex |
| 1cm | 160m × 160m × 20m | Ultra-detailed |
| 5mm | 100m × 100m × 12m | Maximum quality |

### 8x GPU Systems

**8x A100 (640GB total)**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 5cm | 350m × 350m × 40m | Full reef system |
| 2cm | 200m × 200m × 25m | High detail reef |
| 1cm | 130m × 130m × 16m | Research quality |

**8x B200 (1.5TB total)**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 10cm | 700m × 700m × 70m | Coastal region |
| 5cm | 500m × 500m × 60m | Large reef area |
| 2cm | 340m × 340m × 40m | Detailed region |
| 1cm | 220m × 220m × 28m | High-res area |
| 5mm | 140m × 140m × 17m | Ultra-detailed |
| 1mm | 50m × 50m × 5m | Extreme detail |

### 16x GPU Systems

**16x B200 (3TB total)**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 10cm | 1km × 1km × 100m | Full coastal area |
| 5cm | 700m × 700m × 85m | Major reef system |
| 2cm | 480m × 480m × 55m | Detailed coast |
| 1cm | 310m × 310m × 40m | High-res system |
| 5mm | 200m × 200m × 24m | Research grade |
| 1mm | 70m × 70m × 7m | Ultimate detail |

## Computational Performance Estimates

### Single GPU MLUPS (Million Lattice Updates Per Second)
- A200: 800-1000 MLUPS
- A100: 1000-1200 MLUPS
- H100: 1500-2000 MLUPS
- B200: 2500-3500 MLUPS
- MI300X: 2000-2800 MLUPS

### Multi-GPU Scaling
| GPUs | Efficiency | Effective MLUPS |
|------|------------|-----------------|
| 2x | 90% | 1.8x single |
| 4x | 85% | 3.4x single |
| 8x | 80% | 6.4x single |
| 16x | 75% | 12x single |

## Time to Solution

### Simulation Time Steps
For 1 second of physical time at different resolutions:
- 10cm: ~1,000 steps
- 5cm: ~2,000 steps
- 2cm: ~5,000 steps
- 1cm: ~10,000 steps
- 5mm: ~20,000 steps
- 1mm: ~100,000 steps

### Wall Clock Time (1 second physical)
**8x B200 System (20,000 MLUPS effective)**
| Resolution | Domain | Nodes | Time |
|------------|--------|-------|------|
| 5cm | 500×500×60m | 3B | 2.5 hours |
| 2cm | 340×340×40m | 5.8B | 12 hours |
| 1cm | 220×220×28m | 3.4B | 47 hours |

## Comparison with Salvatore

**Salvatore Baseline:**
- 120mm radius spheres = ~240mm spacing
- Limited to small domains on CPU
- Hours for seconds of simulation

**FluidSpace on Big Iron:**
- 50x-240x better resolution
- 1000x-10000x larger domains
- Real-time to minutes for similar timeframes

## Cost-Performance Analysis

### Cloud Costs (Estimated)
- A100: $5-8/hour per GPU
- H100: $8-12/hour per GPU
- B200: $15-20/hour per GPU (projected)

### Cost per Simulation
**Example: 10 seconds of 5cm resolution surf break**
- Domain: 200×200×25m
- 8x A100: ~$320 (8 hours × 8 GPUs × $5)
- 4x B200: ~$240 (3 hours × 4 GPUs × $20)
- 16x B200: ~$160 (30 min × 16 GPUs × $20)

## Recommendations

### For Surf Reef Design (Production)
- **Minimum**: 4x A100 or 2x B200
- **Recommended**: 8x B200
- **Ideal**: 16x B200
- **Resolution**: 2-5cm for design, 1cm for validation

### For Research
- **Minimum**: 1x B200 or MI300X
- **Recommended**: 4x B200
- **Resolution**: 5mm-1cm for detailed physics

### For Real-time Preview
- **Configuration**: 1x H100 or B200
- **Resolution**: 5-10cm
- **Domain**: 100×100×20m sections

## Future Projections

### Next-Gen GPUs (2026-2027)
- 384GB+ HBM4 per GPU
- 10+ TB/s bandwidth
- 5000+ MLUPS single GPU
- Enable: 1mm resolution at reef scale

### Software Optimizations
- Further memory compression (8-10 bytes/node)
- Mixed precision compute
- Adaptive resolution
- AI-accelerated regions

## Cerebras Wafer-Scale Analysis

### Architecture Overview
The Cerebras WSE-3 represents a fundamentally different approach to compute acceleration. Rather than multiple discrete chips, it uses an entire 300mm wafer as a single processor.

**Key Specifications:**
- **On-chip SRAM**: 44GB (vs 80GB HBM on H100)
- **Memory Bandwidth**: 21 PB/s (7,000x H100's 3 TB/s)
- **Compute Cores**: 900,000 AI-optimized cores
- **Fabric Bandwidth**: 214 Pb/s (3,715x H100)
- **Power**: 15-20kW for entire system

### Fluid Simulation Advantages

**Bandwidth Paradise:**
- LBM is memory bandwidth limited
- 21 PB/s enables theoretical 105,000 MLUPS
- No HBM bottleneck - all memory on-chip
- Zero memory hierarchy management

**Spatial Locality:**
- 2D mesh of cores maps naturally to 3D fluid domain
- Nearest-neighbor communication built into fabric
- No PCIe or NVLink bottlenecks
- Perfect for streaming collision operators

### Memory Configuration for LBM

**On-chip Only (44GB):**
- @ 200 bytes/node: 220M nodes
- @ 50 bytes/node: 880M nodes  
- @ 14.5 bytes/node: 3B nodes

**With MemoryX (1.5TB-1.2PB):**
- Weight streaming architecture allows external memory
- Could stream distribution functions from MemoryX
- Keep active working set in 44GB SRAM

### Physical Domain Estimates

**Pure On-chip Configuration (50 bytes/node):**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 10cm | 210m × 210m × 20m | Full surf break |
| 5cm | 130m × 130m × 13m | Detailed break |
| 2cm | 70m × 70m × 8m | High-res section |
| 1cm | 45m × 45m × 5m | Research detail |

**With MemoryX Streaming (1.2PB):**
| Resolution | Domain Size | Notes |
|------------|-------------|-------|
| 10cm | 5km × 5km × 500m | Regional ocean |
| 5cm | 3km × 3km × 300m | Coastal system |
| 2cm | 1.5km × 1.5km × 180m | Large reef complex |
| 1cm | 1km × 1km × 100m | Detailed coast |
| 5mm | 600m × 600m × 70m | Ultra-high res |

### Performance Projections

**Theoretical Peak:**
- 105,000 MLUPS (based on bandwidth)
- Reality: 20,000-50,000 MLUPS expected
- 10-50x faster than GPU clusters

**Time to Solution (1 second physical):**
| Resolution | Domain | Nodes | Time |
|------------|--------|-------|------|
| 5cm | 130×130×13m | 440M | 3 seconds |
| 2cm | 70×70×8m | 490M | 20 seconds |
| 1cm | 45×45×5m | 450M | 90 seconds |

### Programming Considerations

**Advantages:**
- Single memory space - no distribution
- Native dataflow programming model
- Hardware collision/streaming overlap

**Challenges:**
- Custom SDK required
- Limited ecosystem vs CUDA
- Fixed precision requirements
- No dynamic memory allocation

### Cost Analysis

**System Cost:**
- CS-3 system: $2-3M estimated
- Includes cooling, MemoryX units
- 5-year depreciation: $1,400/day

**Cost per Simulation:**
- 10 seconds @ 5cm: ~$2
- 10 seconds @ 1cm: ~$20
- Orders of magnitude cheaper than GPU clusters

### Cerebras vs GPU Comparison

| Metric | 8x B200 | 1x CS-3 | Advantage |
|--------|---------|---------|-----------|
| Memory BW | 64 TB/s | 21 PB/s | 328x CS-3 |
| On-chip Mem | 1.5GB | 44GB | 29x CS-3 |
| Total Memory | 1.5TB | 1.2PB | 800x CS-3 |
| Peak MLUPS | 20,000 | 50,000 | 2.5x CS-3 |
| Power | 5.6kW | 20kW | 3.6x B200 |
| Cost | $3-5M | $2-3M | 1.5x CS-3 |

### Use Cases

**Ideal for Cerebras:**
- Fixed domain sizes
- Uniform resolution
- Long time series
- Research applications
- Real-time preview (small domains)

**Better on GPUs:**
- Adaptive mesh refinement
- Multi-physics coupling
- Irregular geometries
- Production flexibility

### Future Potential

**Next-Gen WSE-4 (Projected):**
- 100GB+ on-chip SRAM
- 50+ PB/s bandwidth
- 3nm process
- Enable full 1mm ocean-scale

**Software Evolution:**
- Direct LBM implementation
- Hardware collision operators
- Streaming boundary conditions
- AI-accelerated turbulence

## Conclusions

1. **Current Generation** (A100/H100): Suitable for 2-5cm production work
2. **Blackwell** (B200): Enables 1cm production, 1-5mm research
3. **Cerebras** (WSE-3): Revolutionary for fixed-domain, bandwidth-limited simulations
4. **Multi-GPU**: Essential for reef-scale simulations with flexibility
5. **AMD Alternative**: MI300X offers compelling price/performance

The combination of modern GPUs and optimized software will enable unprecedented simulation fidelity for artificial reef design and surf dynamics modeling. Cerebras represents a paradigm shift for specific use cases where its architecture aligns with problem requirements.

---

*"From 120mm spheres to 1mm precision - the future of surf is computed"*

*"With Cerebras: From memory bottlenecks to bandwidth paradise"*
