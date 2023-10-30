import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import warnings
warnings.filterwarnings("ignore")
import plotly.express as pt

df=pd.read_csv("GlobalTerrorism.csv",encoding='iso-8859-1')
col=df.columns
print("Columns:")
for column in col:
      print(column)

df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','gname':'Group','country_txt':'Country','region_txt':'Region','provstate':'State','city':'City','latitude':'latitude','longitude':'longitude','attacktype1_txt':'AttackType','target1':'Target','targtype1_txt':'TargetType','weaptype1_txt':'Weapon','nkill':'Kill','nwound':'Wound','summary':'Summary','motive':'Motive'},inplace=True)
df = df[['Year', 'Month', 'Day','Group','Country', 'Region', 'State', 'City', 'latitude', 'longitude', 'AttackType','Target', 'TargetType', 'Weapon', 'Kill', 'Wound', 'Summary', 'Motive']]

df['Wound'] = df['Wound'].fillna(0).astype(int)
df['Kill'] = df['Kill'].fillna(0).astype(int)
df['Casualties'] = df['Kill'] + df['Wound']

df= df.drop(['Kill', 'Wound'], axis=1)
df= df.fillna(0)
plt.figure(figsize=(12, 8))
sea.clustermap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Cluster Map of Attack Attributes')
plt.show()

incidents_per_year = df.groupby('Year').size()
plt.figure(figsize=(10, 6))
plt.plot(incidents_per_year.index, incidents_per_year.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.title('Terrorist Incidents Over the Years')
plt.grid()
plt.show()

yearly_region_attacks = df.groupby(['Year', 'Region']).size().unstack()

country_year_counts = df.groupby(['Country', 'Year']).size().reset_index(name='AttackCount')
max_attack_count = country_year_counts['AttackCount'].max()
plot_data = pd.merge(country_year_counts, df, on=['Country', 'Year'])
fig = pt.choropleth(plot_data,
                    locations='Country',
                    locationmode='country names',
                    color='AttackCount',
                    title='Terrorist Attacks by Country',
                    labels={'AttackCount': 'Number of Attacks'},
                    hover_name='Country',
                    color_continuous_scale='Viridis',
                    animation_frame='Year',
                    range_color=(0, max_attack_count))

fig.show()

yearly_attack_type_counts = df.groupby(['Year', 'AttackType']).size().unstack()
plt.figure(figsize=(12, 6))
yearly_attack_type_counts.plot(kind='area', colormap='hsv', alpha=0.8)
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.title('Temporal Patterns of Attack Types')
plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

yearly_attack_type_counts = df.groupby(['Year', 'AttackType']).size().unstack()
plt.figure(figsize=(12, 6))
yearly_attack_type_counts.plot(kind='area', colormap='hsv', alpha=0.8)
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.title('Temporal Patterns of Attack Types')
plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

top_countries = df['Country'].value_counts().head(10).index
top_countries_data = df[df['Country'].isin(top_countries)]
country_attack_type_counts = top_countries_data.groupby(['Country', 'AttackType']).size().reset_index(name='AttackCount')
pivot_data = country_attack_type_counts.pivot(index='Country', columns='AttackType', values='AttackCount')
plt.figure(figsize=(12, 6))
pivot_data.plot(kind='bar', stacked=False, colormap='Accent')
plt.xlabel('Country')
plt.ylabel('Number of Attacks')
plt.title('Attack Trends Across Countries by Attack Type')
plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.show()