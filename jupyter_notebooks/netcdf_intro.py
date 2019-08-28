import os, sys, glob, time
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt




import xarray as xr


# from mpl_toolkits.basemap import Basemap

file_path = "  /project/projectdirs/dasrepo/mustafa/datasets/Rayleigh_Benard/ result_rb_2d__Ra_2.5e8__Pr_0.71__maxMach_0.1__t_D_max_diffusive_scaling__0.4"


t = time.time()



dataset = Dataset(file_path, mode='r')





ds = xr.open_dataset(file_path)





print(ds)

print(ds.dims ,  ds.data_vars, ds.coords)

print(ds.attrs)


print (dataset.file_format)
print (dataset.dimensions.keys())
print (dataset.variables.keys())
print("time taken: ", time.time() - t)



x = dataset.variables['x'][:]
print("time taken: ", time.time() - t)

y = dataset.variables['y'][:]
print("time taken: ", time.time() - t)

speed_x = dataset.variables['u_x'][:]
print("time taken: ", time.time() - t)

speed_y = dataset.variables['u_y'][:]
print("time taken: ", time.time() - t)


dataset.close()
print("time taken: ", time.time() - t)



# Get some parameters for the Stereographic Projection
lon_0 = x.mean()
lat_0 = y.mean()












# # Get the sea level pressure
# slp = getvar(ncfile, "slp")

# Smooth the sea level pressure since it tends to be noisy near the
# mountains
smooth_slp = smooth2d(slp, 3, cenweight=4)

# Get the latitude and longitude points
lats, lons = latlon_coords(slp)

# Get the cartopy mapping object
cart_proj = get_cartopy(slp)

# Create a figure
fig = plt.figure(figsize=(12,6))
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m",
                             facecolor="none",
                             name="admin_1_states_provinces_shp")
ax.add_feature(states, linewidth=.5, edgecolor="black")
ax.coastlines('50m', linewidth=0.8)

# Make the contour outlines and filled contours for the smoothed sea level
# pressure.
plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), 10, colors="black",
            transform=crs.PlateCarree())
plt.contourf(to_np(lons), to_np(lats), to_np(smooth_slp), 10,
             transform=crs.PlateCarree(),
             cmap=get_cmap("jet"))

# Add a color bar
plt.colorbar(ax=ax, shrink=.98)

# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))

# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

plt.title("Sea Level Pressure (hPa)")

plt.show()


# m = Basemap(width=5000000,height=3500000,
#             resolution='l',projection='stere',\
#             lat_ts=40,lat_0=lat_0,lon_0=lon_0)