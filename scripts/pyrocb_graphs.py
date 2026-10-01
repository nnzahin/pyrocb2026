import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
# from matplotlib.colors import LogNorm

filename = "wrfout_d03_last"
time_index = 0
cloud_level = 10
# cloud_level = 20

with xr.open_dataset(filename) as ds:
    # print(ds)
    # print(ds["Times"].values)

    snapshot = ds.isel(Time=time_index)

    terrain = snapshot["HGT"].load()
    wind_speed = np.hypot(snapshot["U10"], snapshot["V10"]).load()

    # max over height, not one level
    max_w = snapshot["W"].max(dim="bottom_top_stag").load()
    # max_w = snapshot["W"].isel(bottom_top_stag=10).load()

    sr_x = ds.sizes["west_east_subgrid"] // ds.sizes["west_east_stag"]
    sr_y = ds.sizes["south_north_subgrid"] // ds.sizes["south_north_stag"]
    # print(sr_x, sr_y)

    # trim extra fire rows/columns
    fire_slice = {
        "south_north_subgrid": slice(0, -sr_y),
        "west_east_subgrid": slice(0, -sr_x),
    }

    fire_dimensions = {
        "south_north_subgrid": "south_north",
        "west_east_subgrid": "west_east",
    }

    heat_flux = (
        snapshot["FGRNHFX"]
        .isel(fire_slice)
        .rename(fire_dimensions)
        .load() / 1000
    )

    fuel = (
        snapshot["NFUEL_CAT"]
        .isel(fire_slice)
        .rename(fire_dimensions)
        .load()
    )

    cloud_water = (
        snapshot["QCLOUD"]
        .isel(bottom_top=cloud_level)
        .load() * 1000
    )
    # cloud_water = snapshot["QCLOUD"].max("bottom_top").load() * 1000

    cloud_fraction = snapshot["CLDFRA"].max("bottom_top").load() * 100
    cloud_ice = snapshot["QICE"].max("bottom_top").load() * 1000
    tracer = snapshot["tr17_1"].max("bottom_top").load()

    # print(float(snapshot["FGRNHFX"].max()))
    # print(float(snapshot["tr17_1"].max()))

# separate colours for fuel codes
categories, fuel_indices = np.unique(
    np.rint(fuel.values).astype(int),
    return_inverse=True,
)
# print(categories)

fuel_categories = fuel.copy(data=fuel_indices.reshape(fuel.shape))

fuel_colours = plt.get_cmap("tab20")(
    np.linspace(0, 1, len(categories))
)
# fuel_colours = plt.get_cmap("Set3")(np.linspace(0, 1, len(categories)))

fuel_cmap = ListedColormap(fuel_colours)
fuel_norm = BoundaryNorm(
    np.arange(len(categories) + 1) - 0.5,
    fuel_cmap.N,
)

plots = [
    (
        terrain, "Terrain elevation", "terrain",
        "Elevation (m)", "terrain_elevation.png",
    ),
    (
        wind_speed, "10 m wind speed", "viridis",
        "Speed (m/s)", "wind_speed.png",
    ),
    (
        max_w, "Column-maximum vertical velocity", "magma",
        "Vertical velocity (m/s)", "maximum_vertical_velocity.png",
    ),
    (
        heat_flux, "Fire heat release", "inferno",
        "Ground-fire heat flux (kW/m²)", "fire_heat_flux.png",
    ),
    (
        fuel_categories, "Fuel-category distribution", fuel_cmap,
        "Fuel-category code", "fuel_categories.png",
    ),
    (
        cloud_water, f"Cloud water — model level {cloud_level}", "Blues",
        "Cloud-water mixing ratio (g/kg)", "cloud_water.png",
    ),
    (
        cloud_fraction, "Column-maximum cloud fraction", "Blues",
        "Cloud fraction (%)", "cloud_fraction.png",
    ),
    (
        cloud_ice, "Column-maximum cloud ice", "Purples",
        "Cloud-ice mixing ratio (g/kg)", "cloud_ice.png",
    ),
    (
        tracer, "Column-maximum tr17_1 tracer", "inferno",
        "tr17_1 (dimensionless)", "tracer.png",
    ),
]

plot_options = {
    "fire_heat_flux.png": {"vmin": 0},
    "fuel_categories.png": {"norm": fuel_norm},
    "cloud_water.png": {"vmin": 0},
    "cloud_fraction.png": {"vmin": 0, "vmax": 100},
    "cloud_ice.png": {"vmin": 0},
    "tracer.png": {"vmin": 0},
}

# optional fixed scale for comparing times
# plot_options["wind_speed.png"] = {"vmin": 0, "vmax": 16}
# plot_options["maximum_vertical_velocity.png"] = {"vmin": 0, "vmax": 7}

fire_plots = {"fire_heat_flux.png", "fuel_categories.png"}

for field, title, cmap, label, output_filename in plots:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    # fig, ax = plt.subplots(figsize=(7, 6))

    colourbar_options = {"label": label}
    if output_filename == "fuel_categories.png":
        colourbar_options["ticks"] = np.arange(len(categories))

    image = field.plot(
        ax=ax,
        x="west_east",
        y="south_north",
        cmap=cmap,
        cbar_kwargs=colourbar_options,
        **plot_options.get(output_filename, {}),
    )

    if output_filename == "fuel_categories.png":
        image.colorbar.ax.set_yticklabels(
            [str(category) for category in categories]
        )

    grid_name = "fire" if output_filename in fire_plots else "atmospheric"

    # add timestamp
    ax.set_title(f"{title}")
    # ax.set_title(title)

    ax.set_xlabel(f"West–east {grid_name} grid index")
    ax.set_ylabel(f"South–north {grid_name} grid index")
    ax.set_aspect("equal")

    fig.savefig(output_filename, dpi=200, bbox_inches="tight")
    # fig.savefig("early_" + output_filename, dpi=200, bbox_inches="tight")
    # plt.close(fig)

plt.show()
