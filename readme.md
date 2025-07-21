
Simulation data format/data lake
	comprehensive system for ingestion of massive datasets from simulations (currently fluid simulation, but probably also, fire, smoke, gas, thermo, leave stubs for these)
		raw simulation data (velocities, pressure etc. please research these), for different simulation types (lbm currently ONLY, sph, mpm, fem etc.)
		condensed info, i.e. isosurface, vorticity etc (research into what is currently common etc.). I eventually want a format with and index, and nanite/meshlet like functionality for quick initial display, but successive refinement. look into bevy's meshlet implementation, and if that is applicable, and related technologies.
		we want to be able download index data, from say s3. and that contains indexes into the different datasets.
		consideration that a simulation will have 'moments' in time, moving linearly.
		(advanced, down the road)spatial aware storage, so that the simulation viewer can indicate which part is being viewed/critical, and only that part is streamed. think google maps tiling/cesium. but full-3d
		Curent technologies, and approaches. links etc.

		Architectural choices for ingestion (channels/crossbeam/parquet/timeseries/avro (?). cross-language simplicity)
		GPU driven data-munging. i.e. mesh simplification, heirarchical data structures, and other optimizations for efficient processing and visualization.
		Deep dive into open source simulations softwares (genesys, openfoam, etc)
		Look into how openvdb does stuff
		Look into multi-resolution data structures, especially those in taichi & genesys


Curernt simulation memory formats.
current storage uses fp16 & fp32 & fp64.
There is alot of wasted bits storing things like rotation (normalized (0-1, 0-1, 0.1)). analysis of wasted bits for: rotation, vectors(simulations of say fluid will only have velocity up to s certiain point, like 2.0m/s, or something)
look into fibonnacci sphere for storing rotation as an index. needs to be two way mapping between index and rotation.
