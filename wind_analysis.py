# %%
# Imports
import os
os.chdir("/GWSPH/groups/anenberggrp/onawaz/projects/TEMPO_LUR")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tempo_lur.tempo_lur.config import *
from scipy import fftpack
import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import numpy as np
import pandas as pd
import scienceplots
import scipy
import seaborn as sns
import xarray as xr

# %%
# Constants
plt.style.use(['science','no-latex'])
aggs = {"no2":"mean","tempo":"mean"}
palette = ["#7570b3","#d95f02","#1b9e77","#e7298a","#66a61e"]
g = ["AQS ID","no2","tempo","Latitude","Longitude"]
proj = ccrs.PlateCarree()
aqs_args = {"color":palette[0],"marker":"s","linewidth":3,"markersize":7,"label":aqs_label,"linestyle":"--"}
tempo_args = {"color":palette[1],"marker":"s","linewidth":3,"markersize":7,"label":tempo_label,"linestyle":"--"}

slant = False
hours = list(range(8,19))
#months = [(2024,6)]
months = [(2023,8),(2023,9),(2023,10),(2023,11),(2023,12),(2024,1),(2024,2),(2024,3),(2024,4),(2024,5),(2024,6),(2024,7),(2024,8)]
#months = [(2023,8),(2023,12),(2024,1),(2024,2),(2024,6),(2024,7),(2024,8)]


# %%
# Functions
def load_files():
    if slant:
        aqs = pd.read_csv(slant_file)
    else:
        aqs = pd.read_csv(aqs_file)
    aqs = aqs.rename(columns={"Sample Measurement":"no2"})
    near_road = pd.read_excel(road_file)
    aqs["near road"] = aqs["AQS ID"].isin(near_road["AQS ID"])
    aqs = aqs.loc[aqs.tempo.notna()]
    aqs = aqs.loc[aqs.tempo>=0]
    aqs["hour"] = aqs["Time Local"].str[0:2].astype(int)
    aqs["year_month"] = list(zip(aqs.year,aqs.month))
    aqs = aqs.loc[aqs["year_month"].isin(months)].reset_index(drop=True)
    aqs = aqs.loc[aqs["hour"].isin(hours)].reset_index(drop=True)
    aqs.tempo = aqs.tempo / 1e15
    aqs_base = aqs.copy()
    tmp_m, tmp_std = np.mean(aqs.tempo), np.std(aqs.tempo)
    aqs.tempo = (aqs.tempo - tmp_m)/tmp_std
    aqs_m, aqs_std = np.mean(aqs.no2), np.std(aqs.no2)
    aqs.no2 = (aqs.no2 - aqs_m)/aqs_std
    notNearRoad = aqs_base[~aqs_base["near road"]]
    nearRoad = aqs_base[aqs_base["near road"]]
    return aqs_base, aqs, nearRoad, notNearRoad


def add_cartop_features(ax, bounds):
    state_borders = cfeature.NaturalEarthFeature(category='cultural', 
    name='admin_1_states_provinces_lakes', scale='50m')
    ax.add_feature(state_borders, edgecolor='black',facecolor="none", alpha=1, linewidth=0.5, zorder=0)
    ax.add_feature(cfeature.COASTLINE,edgecolor='black',facecolor="none", alpha=1, linewidth=0.5, zorder=0)
    #ax.add_feature(cfeature.LAND,edgecolor='black',facecolor="none", alpha=1, linewidth=0.5, zorder=0)
    ax.set_extent([bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]])
    ax.set_aspect('auto')

def plot_colorbar(ax, settings, cax=None, orientation="horizontal"):
    if cax == None:
        cax = ax.inset_axes([0, 1.25, 1, 0.1])
    norm = mpl.colors.Normalize(vmin=settings["vmin"],vmax=settings["vmax"])
    sm = plt.cm.ScalarMappable(cmap=settings["cmap"], norm=norm)
    sm.set_array([])
    N = 6
    cb = plt.colorbar(sm,cax=cax,  ticks=np.linspace(settings["vmin"],settings["vmax"],N), boundaries=np.arange(settings["vmin"],settings["vmax"]+0.0001,0.1), orientation=orientation)    
    if "label" in settings.keys():
        cb.set_label(settings["label"],labelpad=-50)


# %%
if __name__ == "__main__":
    aqs_base, aqs, nearRoad, notNearRoad = load_files()
    winds_ds = xr.open_dataset(f"{data_dir}raw/05_WIND/era5_wind_2023_2024.nc")
    # winds_ds = winds_ds.sel(latitude=aqs_base.Latitude.unique(),longitude=aqs_base.Longitude.unique(),method="nearest")
    city1 = pd.read_csv(f"{data_dir}processed/ghs_smod/ghs_smod_city_feature_mapping_0_1000.csv")
    city2 = pd.read_csv(f"{data_dir}processed/ghs_smod/ghs_smod_city_feature_mapping_1001_2000.csv")
    city_mapping = gpd.read_file(f"{data_dir}raw/04_GHS_SMOD/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_UC_V1_0/").to_crs("EPSG:9822")
    city = pd.concat([city1,city2])
    z = city_mapping.geometry.centroid
    z = z.to_crs("EPSG:4326")
    city_coords = pd.DataFrame(zip(z.x.astype(float), z.y.astype(float)))
    temp_df = aqs.copy()
    for coord in list(set(zip(aqs_base.Latitude,aqs_base.Longitude))):
        lat, lon = coord
        print(lat)
        ds = winds_ds.sel(latitude=lat,longitude=lon,method="nearest")
        z = ds.to_dataframe().reset_index()
        z["year"] = z.valid_time.astype(str).str[0:4].astype(int)
        z["month"] = z.valid_time.astype(str).str[5:7].astype(int)
        z["hour"] = z.valid_time.astype(str).str[11:16]
        ind = (temp_df.Latitude==lat) & (temp_df.Longitude==lon) 
        df = temp_df.loc[ind]
        for var in ["u10","v10","u100","v100"]:
            mapper = dict(zip(list(zip(z.month,z.year,z.hour)),z[var]))
            test = [*map(mapper.get, zip(df.month,df.year,df["Time GMT"]))]
            df.loc[:,var] = test
            temp_df.loc[ind,var] = df[var]
    # for time in winds_ds.valid_time:
    #     print(time.data)
    #     year, month = int(str(time.data)[0:4]), int(str(time.data)[5:7])
    #     hour = str(time.data)[11:16]
    #     temp = winds_ds.sel(valid_time=time)
    #     ind = (aqs.year==year) & (aqs.month==month) & (aqs["Time GMT"]==hour)
    #     lats =temp_df[ind].Latitude.to_list()
    #     lons =temp_df[ind].Longitude.to_list()        
    #     inds = temp_df[ind].index
    #     if len(lats) == 0:
    #         continue
    #     for lat, lon in zip(lats, lons):
    #         df = temp.sel(latitude=lat,longitude=lon, method="nearest")
    #     # for var in list(temp.variables)[5:]:
    #     #     df_tmp = pd.DataFrame()
    #     #     for lat, lon, i in zip(lats,lons, inds):
    #     #         df = pd.DataFrame({var:temp[var].sel(latitude=lat,longitude=lon,method="nearest").data},index=[i])
    #     #         df_tmp = pd.concat([df_tmp,df])
    #     #         # temp_df.loc[ind,var] = np.diag(temp[var].sel(latitude=lats,longitude=lons,method="nearest").data)
    #     #     temp_df.loc[inds,var] = df_tmp

#%%
# CITY MAPPING
city_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(city_coords[0],city_coords[1]),crs={'init':"EPSG:4326"})
city_points = city_points.to_crs(epsg=3763)
coords = temp_df.groupby(["Latitude","Longitude"]).agg({"AQS ID":"first"}).reset_index()
coords = gpd.GeoDataFrame(coords,geometry=gpd.points_from_xy(coords.Longitude,coords.Latitude),crs={'init':"EPSG:4326"})
coords = coords.to_crs(epsg=3763)
coords["city"] = np.NaN
coords["city_lat"] = np.NaN
coords["city_lon"] = np.NaN

for row in coords.iterrows():
    ind = row[0]
    distances = city_points.distance(row[1].geometry)/1e3 # KM
    min_diff = np.min(distances)
    # if (min_diff > 100) | (min_diff < 25):
    #     continue
    # if (min_diff > 1):
    #     continue
    coords.loc[ind,"city"] = np.argmin(distances)
    coords.loc[ind,"city_lat"] = city_coords.iloc[np.argmin(distances)][1]
    coords.loc[ind,"city_lon"] = city_coords.iloc[np.argmin(distances)][0]
    coords.loc[ind,"city_diff"] = min_diff

    lon_mapper = dict(zip(coords["AQS ID"],coords.city_lon))
    lat_mapper = dict(zip(coords["AQS ID"],coords.city_lat))
    city_mapper = dict(zip(coords["AQS ID"],coords.city))
    diff_mapper = dict(zip(coords["AQS ID"],coords.city_diff))
    temp_df["city_lon"] = [*map(lon_mapper.get, temp_df["AQS ID"])]
    temp_df["city_lat"] = [*map(lat_mapper.get, temp_df["AQS ID"])]
    temp_df["city"] = [*map(city_mapper.get, temp_df["AQS ID"])]
    temp_df["city_dif"] = [*map(diff_mapper.get, temp_df["AQS ID"])]

    temp_df["lon_diff"] = (temp_df.Longitude - temp_df.city_lon)
    temp_df["lat_diff"] = (temp_df.Latitude - temp_df.city_lat)
    temp_df["loc_angle"] = np.rad2deg(np.arctan2(temp_df.lat_diff,temp_df.lon_diff))
    temp_df["wind_angle"] = np.rad2deg(np.arctan2(temp_df.v10,temp_df.u10))
    speed = np.sqrt(temp_df.u10 ** 2 + temp_df.v10 ** 2)

    temp_df["category"] = "non-city"
    ind = (np.abs(temp_df.loc_angle-temp_df.wind_angle) < 60) | (np.abs(temp_df.loc_angle-temp_df.wind_angle) > 300)
    temp_df.loc[ind,"category"] = "downwind"
    ind = (np.abs(temp_df.loc_angle-temp_df.wind_angle) > 120) & (np.abs(temp_df.loc_angle-temp_df.wind_angle) < 240)
    temp_df.loc[ind,"category"] = "upwind"
    ind = ((np.abs(temp_df.loc_angle-temp_df.wind_angle) > 60) & (np.abs(temp_df.loc_angle-temp_df.wind_angle) < 120)) | (np.abs(temp_df.loc_angle-temp_df.wind_angle) > 240) & (np.abs(temp_df.loc_angle-temp_df.wind_angle) < 300)
    temp_df.loc[ind,"category"] = "perpindicular"
    ind = (speed <= 0.5) & (temp_df.city.notna())
    temp_df.loc[ind,"category"] = "stagnant"

    # ind = (temp_df.category == "stagnant")# & (temp_df["near road"])
    # z = temp_df.loc[ind]
    # x,y = z.no2, z.tempo
    # _, _, r_value, _, std_err = scipy.stats.linregress(x, y)
    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.plot(x,y,'r.')
    # ax.text(0,9,f"{r_value * r_value:0.2f}")
    # ax.text(0,8,len(z))
    # ax.set_xlim([-2,10])
    # ax.set_ylim([-2,10])

    # fig, ax = plt.subplots(figsize=(6,3))
    # diurnal = z.groupby("Time Local").agg(aggs)
    # ax.plot(diurnal.no2,color="Purple")
    # ax.plot(diurnal.tempo,color="Orange")

# %%
# %%
# Create Figure (Wind)

bins = [0,0.25,0.5,1,1.5,2,3,5]
temp_df["speed"] = np.sqrt(temp_df.u10 ** 2 + temp_df.v10 ** 2)
temp_df['bin'] = pd.cut(temp_df['speed'], bins=bins, include_lowest=True, right=False)
temp_df['bin'] = temp_df['bin'].apply(lambda x: x.left)
z = temp_df.loc[temp_df.bin.notna()]
z = z.loc[(z.city_dif < 25) & (z.city_dif < 50)]
aqs_means = z.groupby("bin").agg({"no2":"median"}).reset_index(drop=True)
temp_means = z.groupby("bin").agg({"tempo":"median"}).reset_index(drop=True)

cats =["stagnant","downwind","perpindicular","upwind"]
corrs = pd.DataFrame()
bins = []
for i,bin in enumerate(np.unique(z.bin)):
    for cat in cats:
        temp = z.loc[(z.bin == bin) & (z.category == cat)]
        if len(temp) <= 10:
            corrs.loc[i,cat] = np.NaN
        else:
            _, _, r_value, _, std_err = scipy.stats.linregress(temp.no2, temp.tempo)
            corrs.loc[i,cat] = r_value ** 2
            #corrs.append(r_value **2)
    bins.append(bin)

# Draw a nested boxplot to show bills by day and time
bbox = {"facecolor":"white","alpha":0.6,"pad":0}
fig = plt.figure(figsize=(5,5),dpi=300)
ax=fig.add_subplot(111, label="1")
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="no2",y="bin",data=z,showfliers=False,positions=np.arange(0,len(bins))-0.2,width=0.3,color=palette[0],boxprops=dict(alpha=.5)).invert_yaxis()

sns.boxplot(x="tempo",y="bin",data=z,showfliers=False,positions=np.arange(0,len(bins))+0.2,width=0.3,color=palette[1],boxprops=dict(alpha=.5)).invert_yaxis()
ax.text(-1.9,6.1,"TEMPO",fontweight="bold",color=palette[1],bbox=bbox)
ax.text(-1.9,5.7,"AQS",fontweight="bold",color=palette[0],bbox=bbox)
ax.text(4.4,6.1,"Perpindicular",fontweight="bold",color=palette[2],bbox=bbox,ha="right")
ax.text(4.4,5.7,"Downwind",fontweight="bold",color=palette[3],bbox=bbox,ha="right")
ax.text(4.4,5.3,"Upwind",fontweight="bold",color="orange",bbox=bbox,ha="right")
ax.text(4.4,1.2,"Stagnant",fontweight="bold",color="red",bbox=bbox,ha="right")

ax.plot(aqs_means,np.arange(0,len(bins)),'--',marker=".",color=palette[0],linewidth=2,markersize=10)
ax.plot(temp_means,np.arange(0,len(bins)),'--',marker=".",color=palette[1],linewidth=2,markersize=10) 

ax.set_xlabel("Standardized $NO_2$", color="k")
ax.set_ylabel("Wind Speed Bin (m/s)", color="k")
ax.tick_params(axis='x', colors="k")

ax.set_xlim([-2,4.5])
ax.set_ylim([-1,len(bins)])
ax.grid("on",linestyle="--",alpha=0.5)
#ax.legend(["AQS","TEMPO"])

ax2=fig.add_subplot(111, label="2", frame_on=False)
ax2.plot(corrs["stagnant"],np.arange(0,len(bins)),'-',color="red",marker=".",markersize=20,linewidth=3)
ax2.plot(corrs["perpindicular"],np.arange(0,len(bins)),color=palette[2],marker=".",markersize=20,linewidth=3)
ax2.plot(corrs["downwind"],np.arange(0,len(bins)),color=palette[3],marker=".",markersize=20,linewidth=3)
ax2.plot(corrs["upwind"],np.arange(0,len(bins)),color="orange",marker=".",markersize=20,linewidth=3)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top') 
ax2.set_xticks([0,0.25,0.5,0.75,1])
ax2.set_xlim([0,1])
ax2.set_ylim([-1,len(bins)])
ax2.set_yticks([])
ax2.set_xlabel("Correlation ($R^2$)", color="red")
ax2.tick_params(axis='x', colors="red")
#for i,corr in enumerate(corrs):
    #ax2.text(corr+0.05,i-0.1,f"{corr:0.2}",color='r')
ax2.set_xlim([-1,1])
#ax2.grid("on",linestyle="--",color="red",alpha=0.5)
ax.text(-3,7,"a",fontsize=20)





# category = "stagnant"
# cat_name = "Stagnant (between <25 km)"
# fig = plt.figure(figsize=(3, 7),dpi=100)
# gs = gridspec.GridSpec(4, 1, height_ratios=[2, 0., 1, 1],hspace=0.3)
# ind = (temp_df.category == category)
# z = temp_df.loc[ind]
# q = z["city_dif"].quantile(np.arange(10) / 10)
# z["q"] = z["city_dif"].apply(lambda x : q.index[np.searchsorted(q, x, side='right')-1])

# for q in np.linspace(0.1,1,10):
#     temp = z.loc[z.q==q]
#     x = z.no2
#     y = z.tempo
#     _, _, r_value, _, std_err = scipy.stats.linregress(x, y)


# # Scatter
# ax = fig.add_subplot(gs[0,0])
# nnr = z.loc[~z["near road"]]
# x,y = nnr.no2, nnr.tempo
# _, _, r_value, _, std_err = scipy.stats.linregress(x, y)
# N, r2 = len(x), r_value*r_value
# statstr = f"Not Road\n$R^2$={r2:0.2f}\nN={N:0.0f}"
# bbox = {"facecolor":"white","edgecolor":"black","linewidth":0.5}
# ax.text(6 * 0.90, 7, statstr, ha="right", bbox=bbox, color="#66a61e")
# ax.plot(x,y,'.',color="#66a61e")

# nr = z.loc[z["near road"]]
# x,y = nr.no2, nr.tempo
# _, _, r_value, _, std_err = scipy.stats.linregress(x, y)
# N, r2 = len(x), r_value*r_value
# statstr = f"Near Road\n$R^2$={r2:0.2f}\nN={N:0.0f}"
# bbox = {"facecolor":"white","edgecolor":"black","linewidth":0.5}
# ax.text(2 * 0.90, 7, statstr, ha="right", bbox=bbox, color="#e7298a")
# ax.plot(x,y,'.',color="#e7298a")
# ax.grid("on",linestyle="--",alpha=0.5)
# ax.plot(np.linspace(-2,10),np.linspace(-2,10),'k--')
# ax.set_xlabel("AQS $NO_2$ (Standardized)")
# ax.set_ylabel("TEMPO $NO_2$ (Standardized)")
# ax.set_xlim([-2,10])
# ax.set_ylim([-2,10])
# ax.set_title(cat_name)

# # Map
# ax = fig.add_subplot(gs[2,0],projection=proj)
# spatial = z.groupby("AQS ID").agg({"month":"count","Latitude":"first","Longitude":"first","near road":"first"})

# gdf = gpd.GeoDataFrame(spatial,geometry=gpd.points_from_xy(spatial.Longitude,spatial.Latitude))

# gdf.loc[gdf["near road"]].plot(column="month",ax=ax,cmap="cubehelix_r",edgecolor="k",marker="s",markersize=20)
# gdf.loc[~gdf["near road"]].plot(column="month",ax=ax,cmap="cubehelix_r",edgecolor="k",marker=".",markersize=100)
# add_cartop_features(ax,bounds)
# lab = "# of Observations"
# settings = {"vmin":1,"vmax":36,"cmap":"cubehelix_r","label":lab}
# cax = ax.inset_axes([-0.15, 0, 0.03, 1])
# plot_colorbar(ax, settings,cax = cax,orientation="vertical")

# # Diurnal
# ax = fig.add_subplot(gs[3,0])
# diurnal = z.groupby("Time Local").agg(aggs)
# ax.plot(diurnal.no2,**aqs_args)
# ax.plot(diurnal.tempo, **tempo_args)
# ax.legend(["AQS","TEMPO"],loc=2,bbox_to_anchor=(0.5,0.9,0.1,0.1))
# ax.grid("on",linestyle="--",alpha=0.5)
# ax.set_ylabel("$NO_2$ (Standardized)")
# ax.set_xlabel("Hour of Day (LT)")
# for label in ax.xaxis.get_ticklabels()[::2]:
#     label.set_visible(False)

#%%
#BLH

z = xr.open_dataset("/GWSPH/groups/anenberggrp/onawaz/projects/TEMPO_LUR/data/raw/06_ERA5/era5_predictor_variables.nc")
blh = z.blh
temp_df_blh = aqs.copy()
temp_df_blh["blh"] = np.nan
for time in blh.time:
    year, month = int(str(time.data)[0:4]), int(str(time.data)[5:7])
    hour = str(time.data)[11:16]
    temp = blh.sel(time=time)
    #temp_df_blh = aqs.loc[(aqs.year==year) & (aqs.month==month)]
    ind = (aqs.year==year) & (aqs.month==month) & (aqs["Time GMT"]==hour)
    lats =temp_df_blh[ind].Latitude.to_list()
    lons =temp_df_blh[ind].Longitude.to_list()
    if len(lats) == 0:
        continue
    temp_df_blh.loc[ind,"blh"] = np.diag(temp.sel(latitude=lats,longitude=lons,expver=1,method="nearest").data)

#%%# Create Figure (blh)


z = temp_df_blh.copy()
#z = z.loc[z["month"].isin([9,10,11])].reset_index(drop=True)
z = z.loc[z.Longitude < -90]
z = z.loc[z.Latitude > 35]
z['bin'] = pd.cut(z['blh'], bins=10, include_lowest=True, right=False)
z['bin'] = z['bin'].apply(lambda x: np.int64(x.left))
z = z.loc[z.bin.notna()]
aqs_means = z.groupby("bin").agg({"no2":"median"}).reset_index(drop=True)
temp_means = z.groupby("bin").agg({"tempo":"median"}).reset_index(drop=True)

corrs = []
bins = []
for bin in np.unique(z.bin):
    temp = z.loc[z.bin == bin]
    _, _, r_value, _, std_err = scipy.stats.linregress(temp.no2, temp.tempo)
    corrs.append(r_value **2)
    bins.append(bin)

# Draw a nested boxplot to show bills by day and time
fig = plt.figure(figsize=(5,5))
ax=fig.add_subplot(111, label="1")
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="no2",y="bin",data=z,showfliers=False,positions=np.arange(0,10)-0.2,width=0.3,color=palette[0],boxprops=dict(alpha=.5)).invert_yaxis()

sns.boxplot(x="tempo",y="bin",data=z,showfliers=False,positions=np.arange(0,10)+0.2,width=0.3,color=palette[1],boxprops=dict(alpha=.5)).invert_yaxis()
ax.text(-0.3,9,"TEMPO",fontweight="bold",color=palette[1])
ax.text(-0.3,8.5,"AQS",fontweight="bold",color=palette[0])

ax.plot(aqs_means,np.arange(0,10),'--',marker=".",color=palette[0],linewidth=2,markersize=10)
ax.plot(temp_means,np.arange(0,10),'--',marker=".",color=palette[1],linewidth=2,markersize=10)

ax.set_xlabel("Standardized $NO_2$", color="k")
ax.set_ylabel("Boundary Layer Height Bin", color="k")
ax.tick_params(axis='x', colors="k")

ax.set_xlim([-2,5])
ax.grid("on",linestyle="--",alpha=0.5)
#ax.legend(["AQS","TEMPO"])

ax2=fig.add_subplot(111, label="2", frame_on=False)
ax2.plot(corrs,np.arange(0,10),'r-',marker=".",markersize=20,linewidth=3)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top') 
ax2.set_xticks([0,0.25,0.5,0.75,1])
ax2.set_xlim([0,1])
ax2.set_ylim([-1,10])
ax2.set_yticks([])
ax2.set_xlabel("Correlation ($R^2$)", color="red")
ax2.tick_params(axis='x', colors="red")
for i,corr in enumerate(corrs):
    ax2.text(corr+0.05,i-0.1,f"{corr:0.2}",color='r')
ax2.set_xlim([-1,1])


# sns.boxplot(x="no2", y="bin",data=temp_df, positions=np.array(range(temp_df.bin.nunique()))*2.0-0.5, showfliers=False, ax=ax).invert_yaxis()
# sns.boxplot(x="tempo", y="bin",data=temp_df, positions=np.array(range(temp_df.bin.nunique()))*2.0+0.5, showfliers=False, ax=ax).invert_yaxis()
# sns.despine(offset=10, trim=True)
# ax.set_ylim([-1,22])
# ax.set_yticks(np.array(range(temp_df.bin.nunique()))*2.0+0.5)



# zmin, zmax = 200, 2000
# cat_name = f"BLH between {zmin}m & {zmax}m"
# ind = (temp_df.blh < zmax) & (temp_df.blh > zmin)
# fig = plt.figure(figsize=(3, 7),dpi=100)
# gs = gridspec.GridSpec(4, 1, height_ratios=[2, 0., 1, 1],hspace=0.3)
# z = temp_df.loc[ind]

# # Scatter
# ax = fig.add_subplot(gs[0,0])
# nnr = z.loc[~z["near road"]]
# x,y = nnr.no2, nnr.tempo
# _, _, r_value, _, std_err = scipy.stats.linregress(x, y)
# N, r2 = len(x), r_value*r_value
# statstr = f"Not Road\n$R^2$={r2:0.2f}\nN={N:0.0f}"
# bbox = {"facecolor":"white","edgecolor":"black","linewidth":0.5}
# ax.text(6 * 0.90, 7, statstr, ha="right", bbox=bbox, color="#66a61e")
# ax.plot(x,y,'.',color="#66a61e")

# nr = z.loc[z["near road"]]
# x,y = nr.no2, nr.tempo
# _, _, r_value, _, std_err = scipy.stats.linregress(x, y)
# N, r2 = len(x), r_value*r_value
# statstr = f"Near Road\n$R^2$={r2:0.2f}\nN={N:0.0f}"
# bbox = {"facecolor":"white","edgecolor":"black","linewidth":0.5}
# ax.text(2 * 0.90, 7, statstr, ha="right", bbox=bbox, color="#e7298a")
# ax.plot(x,y,'.',color="#e7298a")
# ax.grid("on",linestyle="--",alpha=0.5)
# ax.plot(np.linspace(-2,10),np.linspace(-2,10),'k--')
# ax.set_xlabel("AQS $NO_2$ (Standardized)")
# ax.set_ylabel("TEMPO $NO_2$ (Standardized)")
# ax.set_xlim([-2,10])
# ax.set_ylim([-2,10])
# ax.set_title(cat_name)

# # Map
# ax = fig.add_subplot(gs[2,0],projection=proj)
# spatial = z.groupby("AQS ID").agg({"month":"count","Latitude":"first","Longitude":"first","near road":"first"})

# gdf = gpd.GeoDataFrame(spatial,geometry=gpd.points_from_xy(spatial.Longitude,spatial.Latitude))

# gdf.loc[gdf["near road"]].plot(column="month",ax=ax,cmap="cubehelix_r",edgecolor="k",marker="s",markersize=20)
# gdf.loc[~gdf["near road"]].plot(column="month",ax=ax,cmap="cubehelix_r",edgecolor="k",marker=".",markersize=100)
# add_cartop_features(ax,bounds)
# lab = "# of Observations"
# settings = {"vmin":1,"vmax":36,"cmap":"cubehelix_r","label":lab}
# cax = ax.inset_axes([-0.15, 0, 0.03, 1])
# plot_colorbar(ax, settings,cax = cax,orientation="vertical")

# # Diurnal
# ax = fig.add_subplot(gs[3,0])
# diurnal = z.groupby("Time Local").agg(aggs)
# ax.plot(diurnal.no2,**aqs_args)
# ax.plot(diurnal.tempo, **tempo_args)
# ax.legend(["AQS","TEMPO"],loc=2,bbox_to_anchor=(0.5,0.9,0.1,0.1))
# ax.grid("on",linestyle="--",alpha=0.5)
# ax.set_ylabel("$NO_2$ (Standardized)")
# ax.set_xlabel("Hour of Day (LT)")
# for label in ax.xaxis.get_ticklabels()[::2]:
#     label.set_visible(False)










# # Create Figure (Stagnant)
# fig = plt.figure(figsize=(6, 8),dpi=300)
# gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
# ind = (temp_df.category == "stagnant")
# z = temp_df.loc[ind]


# # Map
# ax = fig.add_subplot(gs[0,:],projection=proj)
# spatial = z.groupby("AQS ID").agg({"month":"count","Latitude":"first","Longitude":"first","near road":"first"})

# gdf = gpd.GeoDataFrame(spatial,geometry=gpd.points_from_xy(spatial.Longitude,spatial.Latitude))

# gdf.loc[gdf["near road"]].plot(column="month",ax=ax,cmap="cubehelix_r",edgecolor="k",marker=".",markersize=150)
# gdf.loc[~gdf["near road"]].plot(column="month",ax=ax,cmap="cubehelix_r",edgecolor="k",marker="s",markersize=20)
# add_cartop_features(ax,bounds)
# lab = "# of Observations"
# settings = {"vmin":1,"vmax":36,"cmap":"cubehelix_r","label":lab}
# plot_colorbar(ax, settings)

# # Scatter
# ax = fig.add_subplot(gs[1,:])
# x,y = z.no2, z.tempo
# _, _, r_value, _, std_err = scipy.stats.linregress(x, y)
# ax.plot(x,y,'r.')
# ax.text(0,9,f"{r_value * r_value:0.2f}")
# ax.text(0,8,len(z))
# ax.set_xlim([-2,10])
# ax.set_ylim([-2,10])
