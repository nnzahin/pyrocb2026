import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

filename = "wrfout_d03_last"
time_index = 0
y_index = 150
# y_index = 100
# y_index = 200

with xr.open_dataset(filename) as ds:
    snapshot = ds.isel(Time=time_index)
    # print(snapshot["Times"].values)

    # one row, all heights and x positions
    w = (
        snapshot["W"]
        .isel(south_north=y_index)
        .transpose("bottom_top_stag", "west_east")
        .values
    )

    height = (
        (
            snapshot["PH"].isel(south_north=y_index)
            + snapshot["PHB"].isel(south_north=y_index)
        )
        .transpose("bottom_top_stag", "west_east")
        .values / 9.81 / 1000
    )

    terrain = (
        snapshot["HGT"]
        .isel(south_north=y_index)
        .values / 1000
    )

    dx = float(ds.attrs["DX"])
    distance = np.arange(w.shape[1]) * dx / 1000

    time_values = np.asarray(snapshot["Times"].values).reshape(-1)
    if time_values.dtype.kind == "S":
        timestamp = b"".join(time_values.tolist()).decode().strip()
    else:
        timestamp = "".join(time_values.astype(str)).strip()

# horizontal coordinates at every vertical level
x = np.broadcast_to(distance, w.shape)
# print(w.shape, height.shape)

# same colour range for rising and sinking air
w_limit = max(float(np.nanmax(np.abs(w))), 0.1)
# w_limit = 10  # use the same limit when comparing different times

levels = np.linspace(-w_limit, w_limit, 41)

fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
# fig, ax = plt.subplots(figsize=(10, 5))

image = ax.contourf(
    x,
    height,
    w,
    levels=levels,
    cmap="RdBu_r",
    extend="both",
)

fig.colorbar(
    image,
    ax=ax,
    label="Vertical velocity (m/s)",
)

# optional contour separating rising and sinking air
# ax.contour(x, height, w, levels=[0], colors="black", linewidths=0.5)

ax.fill_between(
    distance,
    0,
    terrain,
    color="dimgray",
    label="Terrain",
    zorder=3,
)

ax.plot(distance, terrain, color="black", linewidth=0.8, zorder=4)

ax.set_title(
    f"d03 vertical velocity — south–north index {y_index}\n"
    f"{timestamp}"
)
ax.set_xlabel("Nominal distance along west–east grid slice (km)")
ax.set_ylabel("Height above sea level (km)")

ax.set_xlim(distance[0], distance[-1])
ax.set_ylim(0, np.nanmax(height))
# ax.set_ylim(0, 15)
# ax.set_xlim(20, 40)

ax.legend(loc="upper right")
# ax.grid(alpha=0.2)

fig.savefig(
    "vertical_velocity_cross_section.png",
    dpi=200,
    bbox_inches="tight",
)
# plt.close(fig)

plt.show()
