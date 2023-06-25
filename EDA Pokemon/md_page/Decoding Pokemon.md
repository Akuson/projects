##### Soruce for Dataset: https://www.kaggle.com/datasets/mariotormo/complete-pokemon-dataset-updated-090420

##### Importing dataset and modules required for Analysis:


```python
# Importing libraries and database:

import pandas as pd
pd.plotting.register_matplotlib_converters()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_theme()

pokE_path = "./pokedex_(Update_04.21).csv" 
pokedex = pd.read_csv(pokE_path)

# agains_? columns give Damage Multiplier values.
pokedex.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1045 entries, 0 to 1044
    Data columns (total 51 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Unnamed: 0        1045 non-null   int64  
     1   pokedex_number    1045 non-null   int64  
     2   name              1045 non-null   object 
     3   german_name       1045 non-null   object 
     4   japanese_name     1045 non-null   object 
     5   generation        1045 non-null   int64  
     6   status            1045 non-null   object 
     7   species           1045 non-null   object 
     8   type_number       1045 non-null   int64  
     9   type_1            1045 non-null   object 
     10  type_2            553 non-null    object 
     11  height_m          1045 non-null   float64
     12  weight_kg         1044 non-null   float64
     13  abilities_number  1045 non-null   int64  
     14  ability_1         1042 non-null   object 
     15  ability_2         516 non-null    object 
     16  ability_hidden    813 non-null    object 
     17  total_points      1045 non-null   int64  
     18  hp                1045 non-null   int64  
     19  attack            1045 non-null   int64  
     20  defense           1045 non-null   int64  
     21  sp_attack         1045 non-null   int64  
     22  sp_defense        1045 non-null   int64  
     23  speed             1045 non-null   int64  
     24  catch_rate        1027 non-null   float64
     25  base_friendship   930 non-null    float64
     26  base_experience   925 non-null    float64
     27  growth_rate       1044 non-null   object 
     28  egg_type_number   1045 non-null   int64  
     29  egg_type_1        1042 non-null   object 
     30  egg_type_2        285 non-null    object 
     31  percentage_male   872 non-null    float64
     32  egg_cycles        1044 non-null   float64
     33  against_normal    1045 non-null   float64
     34  against_fire      1045 non-null   float64
     35  against_water     1045 non-null   float64
     36  against_electric  1045 non-null   float64
     37  against_grass     1045 non-null   float64
     38  against_ice       1045 non-null   float64
     39  against_fight     1045 non-null   float64
     40  against_poison    1045 non-null   float64
     41  against_ground    1045 non-null   float64
     42  against_flying    1045 non-null   float64
     43  against_psychic   1045 non-null   float64
     44  against_bug       1045 non-null   float64
     45  against_rock      1045 non-null   float64
     46  against_ghost     1045 non-null   float64
     47  against_dragon    1045 non-null   float64
     48  against_dark      1045 non-null   float64
     49  against_steel     1045 non-null   float64
     50  against_fairy     1045 non-null   float64
    dtypes: float64(25), int64(13), object(13)
    memory usage: 416.5+ KB



```python
# REMOVAL OF ASSUMED 'USELESS' COLUMNS:

# Dropping the un-named index column and German/Japanese names under the assumption 
# that these columns ***do not hold any "useful" relation*** to any other column of the for any given index of the database
# and a Pokemon may be defined completely without considering them in these variables.
pokedex.drop(columns=pokedex.columns[[0,3,4]],inplace=True)
```

##### A view of the Data and a description of Features:


```python
# A view of columns 0 to 13: (Name, Species, Status, Type, Abilities, Height, Weight)
pokedex.iloc[0:8,0:14]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_number</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>height_m</th>
      <th>weight_kg</th>
      <th>abilities_number</th>
      <th>ability_1</th>
      <th>ability_2</th>
      <th>ability_hidden</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>1</td>
      <td>Normal</td>
      <td>Seed Pokémon</td>
      <td>2</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>0.7</td>
      <td>6.9</td>
      <td>2</td>
      <td>Overgrow</td>
      <td>NaN</td>
      <td>Chlorophyll</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>1</td>
      <td>Normal</td>
      <td>Seed Pokémon</td>
      <td>2</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>2</td>
      <td>Overgrow</td>
      <td>NaN</td>
      <td>Chlorophyll</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>1</td>
      <td>Normal</td>
      <td>Seed Pokémon</td>
      <td>2</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>2.0</td>
      <td>100.0</td>
      <td>2</td>
      <td>Overgrow</td>
      <td>NaN</td>
      <td>Chlorophyll</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Mega Venusaur</td>
      <td>1</td>
      <td>Normal</td>
      <td>Seed Pokémon</td>
      <td>2</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>2.4</td>
      <td>155.5</td>
      <td>1</td>
      <td>Thick Fat</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>1</td>
      <td>Normal</td>
      <td>Lizard Pokémon</td>
      <td>1</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>0.6</td>
      <td>8.5</td>
      <td>2</td>
      <td>Blaze</td>
      <td>NaN</td>
      <td>Solar Power</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Charmeleon</td>
      <td>1</td>
      <td>Normal</td>
      <td>Flame Pokémon</td>
      <td>1</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>1.1</td>
      <td>19.0</td>
      <td>2</td>
      <td>Blaze</td>
      <td>NaN</td>
      <td>Solar Power</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Charizard</td>
      <td>1</td>
      <td>Normal</td>
      <td>Flame Pokémon</td>
      <td>2</td>
      <td>Fire</td>
      <td>Flying</td>
      <td>1.7</td>
      <td>90.5</td>
      <td>2</td>
      <td>Blaze</td>
      <td>NaN</td>
      <td>Solar Power</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>Mega Charizard X</td>
      <td>1</td>
      <td>Normal</td>
      <td>Flame Pokémon</td>
      <td>2</td>
      <td>Fire</td>
      <td>Dragon</td>
      <td>1.7</td>
      <td>110.5</td>
      <td>1</td>
      <td>Tough Claws</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# A view of columns 14 to 29: (Pokemon Metrics, Egg Type, Base Experience)
pokedex.iloc[0:8,14:30]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_points</th>
      <th>hp</th>
      <th>attack</th>
      <th>defense</th>
      <th>sp_attack</th>
      <th>sp_defense</th>
      <th>speed</th>
      <th>catch_rate</th>
      <th>base_friendship</th>
      <th>base_experience</th>
      <th>growth_rate</th>
      <th>egg_type_number</th>
      <th>egg_type_1</th>
      <th>egg_type_2</th>
      <th>percentage_male</th>
      <th>egg_cycles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>64.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Grass</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>142.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Grass</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>236.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Grass</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>281.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Grass</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>62.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Dragon</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>405</td>
      <td>58</td>
      <td>64</td>
      <td>58</td>
      <td>80</td>
      <td>65</td>
      <td>80</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>142.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Dragon</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>534</td>
      <td>78</td>
      <td>84</td>
      <td>78</td>
      <td>109</td>
      <td>85</td>
      <td>100</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Dragon</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>634</td>
      <td>78</td>
      <td>130</td>
      <td>111</td>
      <td>130</td>
      <td>85</td>
      <td>100</td>
      <td>45.0</td>
      <td>70.0</td>
      <td>285.0</td>
      <td>Medium Slow</td>
      <td>2</td>
      <td>Dragon</td>
      <td>Monster</td>
      <td>87.5</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# A view of columns 30 to 47: (Damage Multiplier columns)
pokedex.iloc[0:8,30:48]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>against_normal</th>
      <th>against_fire</th>
      <th>against_water</th>
      <th>against_electric</th>
      <th>against_grass</th>
      <th>against_ice</th>
      <th>against_fight</th>
      <th>against_poison</th>
      <th>against_ground</th>
      <th>against_flying</th>
      <th>against_psychic</th>
      <th>against_bug</th>
      <th>against_rock</th>
      <th>against_ghost</th>
      <th>against_dragon</th>
      <th>against_dark</th>
      <th>against_steel</th>
      <th>against_fairy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# A description of the data:
def desc_col(series:pd.Series):
        data = {
                "Count":series.count(),
                "Missing": series.isnull().sum(),
                "Unique Count": series.unique().size,
                "Data Type":series.dtype,
                "Mean": (np.nan if (np.issubdtype(series.dtype,object)) else series.mean()),
                "Deviation": (np.nan if (np.issubdtype(series.dtype,object)) else series.std()),
                "Minimum": (np.nan if (np.issubdtype(series.dtype,object)) else series.min()),
                "Maximum": (np.nan if (np.issubdtype(series.dtype,object)) else series.max()),
        }
        return pd.Series(data)


"""
Analysis that follows is based on the following assumption: 

        The list of 48 variables/features, describes and defines (**NOT NECESSARILY COMPLETELY**)
        any given "Pokemon".(i.e. 2 distinct "Pokemon" may exist which have 
        all 48 features in common, but no "Pokemon" may be completely defined without any of these 48 features.)

        Furthermore, let us assume, for the sake of simplicity of our analysis, that all 'object' type variables 
        are of "categorical" nature. That is to say, the only significant and meaningful property of a Pokemon(by our definition) 
        represented by the values of these variables is their distinct nature. The only "useful" property of the different 
        values of a categorical variable is their "difference"/distinction, with respect to their role played in defining a Pokemon. 
        The exact symbolic representation of them is arbitrary and purely based on convention/preference.
        (One such variable which is exception to this assumption: 'name')

Hence, the question that will forever remain un-answered, despite any form of analysis possible: What is a Pokemon?
"""

pokedex.apply(desc_col,axis="index").T.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Count</th>
      <th>Missing</th>
      <th>Unique Count</th>
      <th>Data Type</th>
      <th>Mean</th>
      <th>Deviation</th>
      <th>Minimum</th>
      <th>Maximum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pokedex_number</td>
      <td>1045</td>
      <td>0</td>
      <td>898</td>
      <td>int64</td>
      <td>440.769378</td>
      <td>262.517231</td>
      <td>1</td>
      <td>898</td>
    </tr>
    <tr>
      <th>1</th>
      <td>name</td>
      <td>1045</td>
      <td>0</td>
      <td>1045</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>generation</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>int64</td>
      <td>4.098565</td>
      <td>2.272788</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>status</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>species</td>
      <td>1045</td>
      <td>0</td>
      <td>652</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>type_number</td>
      <td>1045</td>
      <td>0</td>
      <td>2</td>
      <td>int64</td>
      <td>1.529187</td>
      <td>0.499386</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>18</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>type_2</td>
      <td>553</td>
      <td>492</td>
      <td>19</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>height_m</td>
      <td>1045</td>
      <td>0</td>
      <td>61</td>
      <td>float64</td>
      <td>1.374067</td>
      <td>3.353349</td>
      <td>0.1</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>weight_kg</td>
      <td>1044</td>
      <td>1</td>
      <td>474</td>
      <td>float64</td>
      <td>71.216571</td>
      <td>132.259911</td>
      <td>0.1</td>
      <td>999.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>abilities_number</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>int64</td>
      <td>2.2689</td>
      <td>0.803154</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ability_1</td>
      <td>1042</td>
      <td>3</td>
      <td>213</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ability_2</td>
      <td>516</td>
      <td>529</td>
      <td>127</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ability_hidden</td>
      <td>813</td>
      <td>232</td>
      <td>155</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>total_points</td>
      <td>1045</td>
      <td>0</td>
      <td>216</td>
      <td>int64</td>
      <td>439.35311</td>
      <td>121.992897</td>
      <td>175</td>
      <td>1125</td>
    </tr>
    <tr>
      <th>15</th>
      <td>hp</td>
      <td>1045</td>
      <td>0</td>
      <td>103</td>
      <td>int64</td>
      <td>70.067943</td>
      <td>26.671411</td>
      <td>1</td>
      <td>255</td>
    </tr>
    <tr>
      <th>16</th>
      <td>attack</td>
      <td>1045</td>
      <td>0</td>
      <td>123</td>
      <td>int64</td>
      <td>80.476555</td>
      <td>32.432728</td>
      <td>5</td>
      <td>190</td>
    </tr>
    <tr>
      <th>17</th>
      <td>defense</td>
      <td>1045</td>
      <td>0</td>
      <td>114</td>
      <td>int64</td>
      <td>74.670813</td>
      <td>31.259462</td>
      <td>5</td>
      <td>250</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sp_attack</td>
      <td>1045</td>
      <td>0</td>
      <td>119</td>
      <td>int64</td>
      <td>73.031579</td>
      <td>32.745857</td>
      <td>10</td>
      <td>194</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sp_defense</td>
      <td>1045</td>
      <td>0</td>
      <td>107</td>
      <td>int64</td>
      <td>72.298565</td>
      <td>28.098943</td>
      <td>20</td>
      <td>250</td>
    </tr>
    <tr>
      <th>20</th>
      <td>speed</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>68.807656</td>
      <td>30.210094</td>
      <td>5</td>
      <td>200</td>
    </tr>
    <tr>
      <th>21</th>
      <td>catch_rate</td>
      <td>1027</td>
      <td>18</td>
      <td>36</td>
      <td>float64</td>
      <td>92.739046</td>
      <td>75.878404</td>
      <td>3.0</td>
      <td>255.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>base_friendship</td>
      <td>930</td>
      <td>115</td>
      <td>7</td>
      <td>float64</td>
      <td>64.064516</td>
      <td>21.452532</td>
      <td>0.0</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>base_experience</td>
      <td>925</td>
      <td>120</td>
      <td>180</td>
      <td>float64</td>
      <td>153.716757</td>
      <td>79.28397</td>
      <td>36.0</td>
      <td>608.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>growth_rate</td>
      <td>1044</td>
      <td>1</td>
      <td>7</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>egg_type_number</td>
      <td>1045</td>
      <td>0</td>
      <td>3</td>
      <td>int64</td>
      <td>1.269856</td>
      <td>0.450522</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>egg_type_1</td>
      <td>1042</td>
      <td>3</td>
      <td>16</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>egg_type_2</td>
      <td>285</td>
      <td>760</td>
      <td>12</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>percentage_male</td>
      <td>872</td>
      <td>173</td>
      <td>7</td>
      <td>float64</td>
      <td>54.873853</td>
      <td>20.398192</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>egg_cycles</td>
      <td>1044</td>
      <td>1</td>
      <td>11</td>
      <td>float64</td>
      <td>31.489464</td>
      <td>30.511128</td>
      <td>5.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>against_normal</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.86866</td>
      <td>0.286863</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>against_fire</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>float64</td>
      <td>1.126316</td>
      <td>0.714569</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>against_water</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.050718</td>
      <td>0.609383</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>against_electric</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.035646</td>
      <td>0.643535</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>against_grass</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.001196</td>
      <td>0.74435</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>against_ice</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.675598</td>
      <td>7.685308</td>
      <td>0.0</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>against_fight</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.07512</td>
      <td>0.753649</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>against_poison</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.95311</td>
      <td>0.541238</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>against_ground</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.082297</td>
      <td>0.782683</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>against_flying</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.1689</td>
      <td>0.592145</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>against_psychic</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>0.977273</td>
      <td>0.501934</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>against_bug</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.998086</td>
      <td>0.610411</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>against_rock</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.238278</td>
      <td>0.69656</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>against_ghost</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.01866</td>
      <td>0.568056</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>against_dragon</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.977033</td>
      <td>0.375812</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>against_dark</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.071053</td>
      <td>0.465178</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>against_steel</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.981579</td>
      <td>0.501753</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>against_fairy</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>1.091148</td>
      <td>0.536285</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Data Cleaning, Data Preperation, Feature Imputation and Compression:


```python
# Exploring the 'type_number' column:

# The 'type_number' of a Pokemon, ideally describes the number of 
# Types that a Pokemon has: Either 1(Pure Type) or 2(Mixed Type).
# Let us verify whether this fact holds for the given data.

type_num_count = pokedex['type_number'].value_counts().sort_index()

print(type_num_count.size)
# The 'type_number' column has 2 unique values, possibly to distinguish
# between "Pure Type" and "Mixed Type".

# Check if any Pokemon with 'type_number' 2 has NaN 'type_2' column
print("Does any exception exist:",(pokedex.loc[(pokedex['type_number']==2)])['type_2'].isnull().any())

# Check if any Pokemon with 'type_number' 1 has non-null 'type_2' column
print("Does any exception exist:",((pokedex.loc[(pokedex['type_number']==1)])['type_2'].isnull()==False).any())

# Hence proved that our observation is correct as both answers are False.

# Renaming type_number indices.
type_num_count.index = ["Pure Type","Mixed Type"]
print(type_num_count)
```

    2
    Does any exception exist: False
    Does any exception exist: False
    Pure Type     492
    Mixed Type    553
    Name: type_number, dtype: int64



```python
# Encoding the pokemon Types:
type_encoding = pd.Series(0,index=pokedex["type_1"].unique()) + pd.Series(0,index=pokedex["type_2"].loc[pokedex["type_2"].isnull()==False].unique()) 

type_encoding.iloc[:] = np.arange(1,type_encoding.size+1)
type_encoding["Null"] = 0
type_encoding.sort_values(inplace=True)
print(type_encoding)

# We drop the 'type_number' column as all information contained in it 
# can be deduced from "type_1" and "type_2" columns by:
# type_number(P) = (type_1(P)!=0) + (type_2(P)!=0) ; for all Pokemon(P)
# Hence it is redundant and unnecessary for further analysis.
pokedex.drop(columns=["type_number"],inplace=True,errors="ignore")
```

    Null         0
    Bug          1
    Dark         2
    Dragon       3
    Electric     4
    Fairy        5
    Fighting     6
    Fire         7
    Flying       8
    Ghost        9
    Grass       10
    Ground      11
    Ice         12
    Normal      13
    Poison      14
    Psychic     15
    Rock        16
    Steel       17
    Water       18
    dtype: int64



```python
# Converting columns "type_1" and "type_2" to numerical types:
def map_type(val,table):
    return table.get(val,0)

pokedex["type_1"] = pokedex["type_1"].apply(map_type,table=type_encoding)
pokedex["type_2"] = pokedex["type_2"].apply(map_type,table=type_encoding)

# Let us also encode the Damage multiplier columns accoriding to our Type Encoding(there are 18 columns each for one Type):
# Check from previous description of columns, we see Damage Multipliers are the *last* 18 columns.[from column indices (30 to 47) or (-1 to -18)]

# For each column name, the first 8 characters are to be omitted and the 9th character capitalized to obtain the corresponding matching Type.
# Exception: The column for 'Fighting' Type is named "against_fight", hence it must be changed to "against_fighting" first.
pokedex.rename(columns={"against_fight":"against_fighting"},inplace=True)
pokedex.rename(lambda x: (f"D{type_encoding[x[8:9].upper()+x[9:]]}") if ("against_" in x) else x,axis="columns",inplace=True)
```


```python
# Encoding Egg Types similarly to Pokemon Types:

egg_encoding = pd.Series(0,index=pokedex["egg_type_1"].unique()) + pd.Series(0,index=pokedex["egg_type_2"].unique())
egg_encoding.iloc[:] = np.arange(0,egg_encoding.size)
egg_encoding.rename({np.nan:"Null"},inplace=True)
egg_encoding = egg_encoding.astype("int64")
print(egg_encoding)

pokedex["egg_type_1"] = pokedex["egg_type_1"].apply(map_type,table=egg_encoding)
pokedex["egg_type_2"] = pokedex["egg_type_2"].apply(map_type,table=egg_encoding)

# Similar to the 'type_number' column, it also inidcates the 
# Egg Type , whether Pure(1), Mixed(2) or NULL(0):
# egg_type_number(P) = (egg_type_1(P)!=0) + (egg_type_2(P)!=0) ; for all Pokemon 'P' 
print("Anomalies :",(pokedex.loc[(pokedex["egg_type_number"].astype(int)-((pokedex["egg_type_1"]!=0).astype(int)+(pokedex["egg_type_2"]!=0).astype(int)))!=0]).size)

# Dropping unnecessary columns:
pokedex.drop(columns=["egg_type_number"],inplace=True,errors="ignore")
```

    Null             0
    Amorphous        1
    Bug              2
    Ditto            3
    Dragon           4
    Fairy            5
    Field            6
    Flying           7
    Grass            8
    Human-Like       9
    Mineral         10
    Monster         11
    Undiscovered    12
    Water 1         13
    Water 2         14
    Water 3         15
    dtype: int64
    Anomalies : 0



```python
# Encoding Abilities of Pokemon:
# There are 3 types of abilities: 'ability_1', 'ability_2' and 'ability_hidden'.

# Let us check whether these sets of abilities are intersecting or not:
print("There are %d common values in 'ability_1' and 'ability_2'."%pokedex["ability_1"].loc[pokedex["ability_1"].isin(pokedex["ability_2"])].nunique(dropna=False))
print("There are %d common values in 'ability_hidden' and 'ability_2'."%pokedex["ability_hidden"].loc[pokedex["ability_hidden"].isin(pokedex["ability_2"])].nunique(dropna=False))
print("There are %d common values in 'ability_hidden' and 'ability_1'."%pokedex["ability_hidden"].loc[pokedex["ability_hidden"].isin(pokedex["ability_1"])].nunique(dropna=False))

# As there is no Ability is mutually exclusive of any other Ability, let us consider all abilities together.
ability_encoding = pd.Series(0,pokedex["ability_1"].unique()) + pd.Series(0,pokedex["ability_2"].unique()) + pd.Series(0,pokedex["ability_hidden"].unique())
ability_encoding.iloc[:] = np.arange(ability_encoding.size)
ability_encoding.rename(index={np.nan:"Null"},inplace=True)
ability_encoding = ability_encoding.astype("int64")
print(ability_encoding.head())

pokedex["ability_1"] = pokedex["ability_1"].apply(map_type,table=ability_encoding)
pokedex["ability_2"] = pokedex["ability_2"].apply(map_type,table=ability_encoding)
pokedex["ability_hidden"] = pokedex["ability_hidden"].apply(map_type,table=ability_encoding)

# Similar to the 'type_number' column, it also inidcates the 
# number of abilities, whether 1, 2, or 0:
# abilities_number(P) = (ability_1(P)!=0) + (ability_2(P)!=0) + (ability_hidden(P)!=0) ; for all Pokemon 'P' 
print("Anomalies :",(pokedex.loc[(pokedex["abilities_number"].astype(int)-((pokedex["ability_1"]!=0).astype(int)+(pokedex["ability_2"]!=0).astype(int)+(pokedex["ability_hidden"]!=0).astype(int)))!=0]).size)

# Dropping unnecessary columns:
pokedex.drop(columns=["abilities_number"],inplace=True,errors="ignore")

```

    There are 100 common values in 'ability_1' and 'ability_2'.
    There are 97 common values in 'ability_hidden' and 'ability_2'.
    There are 112 common values in 'ability_hidden' and 'ability_1'.
    Null            0
    Adaptability    1
    Aerilate        2
    Aftermath       3
    Air Lock        4
    dtype: int64
    Anomalies : 0



```python
# Encoding the Status of a Pokemon:

print("The 'status' property of a Pokemon has only",pokedex['status'].nunique(dropna=False),"unique values.")

status_encoding = pd.Series(np.arange(1,5),index=pokedex['status'].unique())
print(status_encoding)

pokedex["status"] = pokedex["status"].apply(map_type,table=status_encoding)
```

    The 'status' property of a Pokemon has only 4 unique values.
    Normal           1
    Sub Legendary    2
    Legendary        3
    Mythical         4
    dtype: int64



```python
# Encoding the 'growth_rate' of a Pokemon:

print("The 'status' property of a Pokemon has only",pokedex["growth_rate"].nunique(dropna=False),"unique values:")
print(pokedex["growth_rate"].unique())

gr_encoding = pd.Series({'Medium Slow':4, 'Medium Fast':5, 'Fast':6, 'Slow':3, 'Fluctuating':2, 'Erratic':1, 'Null':0,})
gr_encoding.sort_values(inplace=True)

pokedex["growth_rate"] = pokedex["growth_rate"].apply(map_type,table=gr_encoding)
```

    The 'status' property of a Pokemon has only 7 unique values:
    ['Medium Slow' 'Medium Fast' 'Fast' 'Slow' 'Fluctuating' 'Erratic' nan]



```python
# Initial analysis of 'pokedex_number':

# Notice that although the pokedex has 1045 indices, there are 
# only 898 unique 'pokedex_number' values. This means multiple 
# pokemon share a common 'pokedex_number'. Let us examine them:

# Stores each unique 'pokedex_number' and the number of entries for it.
poke_nums_dupe = pokedex['pokedex_number'].value_counts()

# The 'pokedex_number' seems to span completely from 1 to 898.
# However we verify this observation:

print(poke_nums_dupe.index.dtype) # int64
print(poke_nums_dupe.index.min()) # Minimum value: 1
print(poke_nums_dupe.index.max()) # Maximum value: 898
print(poke_nums_dupe.size) # Total number of values 898

# Thus considering all values are >=1 and <=898, there can be atmost,
# '(898 - 1) + 1' unique integers, which if there are exactly '(898 - 1) + 1', 
# implies that the whole range from 1 to 898 is present. Thus verified.

# Let us filter for only those Pokedex numbers which have more than 
# 1 entry corresponding to them.
poke_nums_dupe = poke_nums_dupe[poke_nums_dupe!=1]
```

    int64
    1
    898
    898



```python
# Continued Analysis of 'pokedex_number':

# We know a valid 'pokedex_number' is any integer 'x' : 0 < x < 899 
#Display all pokemon with a fixed 'pokedex_number' from 'poke_nums_dupe'
x = poke_nums_dupe.index[np.random.randint(0,poke_nums_dupe.size)]
pokedex[pokedex['pokedex_number']==146] # Highlighting a counter-example.

# Hypothesis:
# All Pokemon having same 'pokedex_number' seem to share:
#   Common 'status' tag
#   Common  'generation' tag -> (Incorrect) [Counter examples found: 199, 79, 145, 146]
#   A common part in their 'name' tags. Let us find these patterns in the names of Pokemon.

# However, whether these properties completely define the 'pokedex_number', I am still unsure.
# Is it true that there exists no 2 known Pokemon, with the same 'status', 'generation' tag, but distinct 'pokedex_number' values?
# Answer: No, counter examples have been found.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>height_m</th>
      <th>weight_kg</th>
      <th>ability_1</th>
      <th>...</th>
      <th>D11</th>
      <th>D8</th>
      <th>D15</th>
      <th>D1</th>
      <th>D16</th>
      <th>D9</th>
      <th>D3</th>
      <th>D2</th>
      <th>D17</th>
      <th>D5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>188</th>
      <td>146</td>
      <td>Moltres</td>
      <td>1</td>
      <td>2</td>
      <td>Flame Pokémon</td>
      <td>7</td>
      <td>8</td>
      <td>2.0</td>
      <td>60.0</td>
      <td>159</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>189</th>
      <td>146</td>
      <td>Galarian Moltres</td>
      <td>8</td>
      <td>2</td>
      <td>Malevolent Pokémon</td>
      <td>2</td>
      <td>8</td>
      <td>2.0</td>
      <td>66.0</td>
      <td>18</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 45 columns</p>
</div>




```python
# Status - Catch Rate

sns.catplot(data=pokedex,x="status",y="catch_rate",aspect=2.5,hue='status',palette=sns.color_palette("viridis",as_cmap=True),jitter=0.34)
plt.xlabel("Status of Pokemon")
plt.ylabel("Catch Rate")
plt.title("Relation between Catch Rate and Status Tag")
plt.show()

# Encoding Catch Rate:

# We give the missing 'catch_rate' values the symbol '0',
pokedex.loc[:,"catch_rate"].fillna(0,inplace=True)

# The exceptional Legendary Pokemon with high Catch Rate:
pokedex[(pokedex["catch_rate"]>250) & (pokedex["status"]==3)].iloc[:,[0,1,2,3,4,5,6,7,21]]

# Observation: All Pokemon which are Status-2 or higher have less than 50 Catch Rate, barring 2 exceptions.
```


    
![png](output_19_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>height_m</th>
      <th>base_experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>933</th>
      <td>800</td>
      <td>Necrozma</td>
      <td>7</td>
      <td>3</td>
      <td>Prism Pokémon</td>
      <td>15</td>
      <td>0</td>
      <td>2.4</td>
      <td>270.0</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>890</td>
      <td>Eternatus</td>
      <td>8</td>
      <td>3</td>
      <td>Gigantic Pokémon</td>
      <td>14</td>
      <td>3</td>
      <td>20.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The Percentage Male of Pokemon:
print(pokedex["percentage_male"].unique())

# The 'percentage_male' of a Pokemon is completely defined for all Pokemon as a real number from 0 to 100.
# Justification for it might be that any given Pokemon can either be "Male" or "Not Male".
# Based on the above interpretation, there cannot exist any value outside the range [0,100].
# Let us fill in all missing "percentage_male" by 0.
pokedex.loc[:,"percentage_male"].fillna(0,inplace=True)
```

    [ 87.5  50.    0.  100.   25.    nan  75. ]



```python
# Egg Cycles - Status

sns.catplot(data=pokedex,x="status",y="egg_cycles",hue="status",palette=sns.color_palette("viridis",as_cmap=True),aspect=2.5,jitter=0.3)
plt.xlabel("Status of Pokemon")
plt.ylabel("Egg Cycles")
plt.title("Relation of Egg Cycles with Status of Pokemon")
```




    Text(0.5, 1.0, 'Relation of Egg Cycles with Status of Pokemon')




    
![png](output_21_1.png)
    



```python
# Approximating missing Egg Cycles values:

# From the above plot, there is a clear indication of the relationship of 'egg_cycles' with the 'status' of a Pokemon.
# Hence, let us replace the missing values with the average, for Pokemon of different Status seperately.
egg_cycles_wrt_status = pokedex.groupby("status").apply(lambda df: pd.Series({"Mean":df["egg_cycles"].mean(),"Deviation":df["egg_cycles"].std()}))
print(egg_cycles_wrt_status)

# Now let us replace the missing values with these averages:
pokedex.loc[pokedex["egg_cycles"].isnull(),"egg_cycles"] = pokedex.loc[pokedex["egg_cycles"].isnull()].apply(lambda s: egg_cycles_wrt_status["Mean"][s["status"]],axis="columns")

# An interesting relation is that ALL Status-3(Legendary) Pokemon have the same Egg-cycle value of 120.
```

                  Mean  Deviation
    status                       
    1        20.965104   6.266540
    2        99.454545  32.484651
    3       120.000000   0.000000
    4       104.666667  33.577839



```python
# Encoding the 'species' values:
species_encoding = pd.Series(0,index=pokedex["species"].unique())
species_encoding.iloc[:] = np.arange(species_encoding.shape[0])

pokedex["species"] = pokedex.loc[:,"species"].apply(lambda v: species_encoding[v])
```


```python
pokedex.apply(desc_col,axis="index").T.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Count</th>
      <th>Missing</th>
      <th>Unique Count</th>
      <th>Data Type</th>
      <th>Mean</th>
      <th>Deviation</th>
      <th>Minimum</th>
      <th>Maximum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pokedex_number</td>
      <td>1045</td>
      <td>0</td>
      <td>898</td>
      <td>int64</td>
      <td>440.769378</td>
      <td>262.517231</td>
      <td>1</td>
      <td>898</td>
    </tr>
    <tr>
      <th>1</th>
      <td>name</td>
      <td>1045</td>
      <td>0</td>
      <td>1045</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>generation</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>int64</td>
      <td>4.098565</td>
      <td>2.272788</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>status</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>int64</td>
      <td>1.219139</td>
      <td>0.651271</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>species</td>
      <td>1045</td>
      <td>0</td>
      <td>652</td>
      <td>int64</td>
      <td>285.697608</td>
      <td>198.263458</td>
      <td>0</td>
      <td>651</td>
    </tr>
    <tr>
      <th>5</th>
      <td>type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>18</td>
      <td>int64</td>
      <td>10.407656</td>
      <td>5.524097</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>type_2</td>
      <td>1045</td>
      <td>0</td>
      <td>19</td>
      <td>int64</td>
      <td>5.041148</td>
      <td>5.827109</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>height_m</td>
      <td>1045</td>
      <td>0</td>
      <td>61</td>
      <td>float64</td>
      <td>1.374067</td>
      <td>3.353349</td>
      <td>0.1</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>weight_kg</td>
      <td>1044</td>
      <td>1</td>
      <td>474</td>
      <td>float64</td>
      <td>71.216571</td>
      <td>132.259911</td>
      <td>0.1</td>
      <td>999.9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ability_1</td>
      <td>1045</td>
      <td>0</td>
      <td>213</td>
      <td>int64</td>
      <td>137.22201</td>
      <td>74.445257</td>
      <td>0</td>
      <td>265</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ability_2</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>70.616268</td>
      <td>87.593765</td>
      <td>0</td>
      <td>262</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ability_hidden</td>
      <td>1045</td>
      <td>0</td>
      <td>155</td>
      <td>int64</td>
      <td>111.911005</td>
      <td>89.341561</td>
      <td>0</td>
      <td>266</td>
    </tr>
    <tr>
      <th>12</th>
      <td>total_points</td>
      <td>1045</td>
      <td>0</td>
      <td>216</td>
      <td>int64</td>
      <td>439.35311</td>
      <td>121.992897</td>
      <td>175</td>
      <td>1125</td>
    </tr>
    <tr>
      <th>13</th>
      <td>hp</td>
      <td>1045</td>
      <td>0</td>
      <td>103</td>
      <td>int64</td>
      <td>70.067943</td>
      <td>26.671411</td>
      <td>1</td>
      <td>255</td>
    </tr>
    <tr>
      <th>14</th>
      <td>attack</td>
      <td>1045</td>
      <td>0</td>
      <td>123</td>
      <td>int64</td>
      <td>80.476555</td>
      <td>32.432728</td>
      <td>5</td>
      <td>190</td>
    </tr>
    <tr>
      <th>15</th>
      <td>defense</td>
      <td>1045</td>
      <td>0</td>
      <td>114</td>
      <td>int64</td>
      <td>74.670813</td>
      <td>31.259462</td>
      <td>5</td>
      <td>250</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sp_attack</td>
      <td>1045</td>
      <td>0</td>
      <td>119</td>
      <td>int64</td>
      <td>73.031579</td>
      <td>32.745857</td>
      <td>10</td>
      <td>194</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sp_defense</td>
      <td>1045</td>
      <td>0</td>
      <td>107</td>
      <td>int64</td>
      <td>72.298565</td>
      <td>28.098943</td>
      <td>20</td>
      <td>250</td>
    </tr>
    <tr>
      <th>18</th>
      <td>speed</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>68.807656</td>
      <td>30.210094</td>
      <td>5</td>
      <td>200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>catch_rate</td>
      <td>1045</td>
      <td>0</td>
      <td>36</td>
      <td>float64</td>
      <td>91.141627</td>
      <td>76.183957</td>
      <td>0.0</td>
      <td>255.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>base_friendship</td>
      <td>930</td>
      <td>115</td>
      <td>7</td>
      <td>float64</td>
      <td>64.064516</td>
      <td>21.452532</td>
      <td>0.0</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>base_experience</td>
      <td>925</td>
      <td>120</td>
      <td>180</td>
      <td>float64</td>
      <td>153.716757</td>
      <td>79.28397</td>
      <td>36.0</td>
      <td>608.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>growth_rate</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>int64</td>
      <td>4.196172</td>
      <td>1.084895</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>23</th>
      <td>egg_type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>16</td>
      <td>int64</td>
      <td>7.164593</td>
      <td>3.799033</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>24</th>
      <td>egg_type_2</td>
      <td>1045</td>
      <td>0</td>
      <td>12</td>
      <td>int64</td>
      <td>2.873684</td>
      <td>4.926989</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>25</th>
      <td>percentage_male</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>45.789474</td>
      <td>27.631585</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>egg_cycles</td>
      <td>1045</td>
      <td>0</td>
      <td>11</td>
      <td>float64</td>
      <td>31.479392</td>
      <td>30.49825</td>
      <td>5.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>D13</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.86866</td>
      <td>0.286863</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>D7</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>float64</td>
      <td>1.126316</td>
      <td>0.714569</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>D18</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.050718</td>
      <td>0.609383</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>D4</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.035646</td>
      <td>0.643535</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>D10</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.001196</td>
      <td>0.74435</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>D12</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.675598</td>
      <td>7.685308</td>
      <td>0.0</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>D6</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.07512</td>
      <td>0.753649</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>D14</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.95311</td>
      <td>0.541238</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>D11</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.082297</td>
      <td>0.782683</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>D8</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.1689</td>
      <td>0.592145</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>D15</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>0.977273</td>
      <td>0.501934</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>D1</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.998086</td>
      <td>0.610411</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>D16</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.238278</td>
      <td>0.69656</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>D9</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.01866</td>
      <td>0.568056</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D3</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.977033</td>
      <td>0.375812</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>D2</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.071053</td>
      <td>0.465178</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>D17</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.981579</td>
      <td>0.501753</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>D5</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>1.091148</td>
      <td>0.536285</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Exploratory Analysis of the Data:


```python
# Distribution of Pokemon among various Types:

type_colors = pd.Series(["#ffffff","#539620","#6b3a20","#3c0069","#f6fa02","#f995fc","#ab0f1a","#fa6e02",
                         "#73c8fa","#3a3a3b","#41f24d","#d4af37","#02fadd","#e5f27e","#aa10c2",
                         "#fa287c","#696b6e","#b5b6ba","#023ef2"],index=type_encoding.index)


pure_type_count = (pokedex[pokedex["type_2"]==0].groupby("type_1").apply(lambda df: df.shape[0])).rename(lambda x: type_encoding.index[x]).reset_index()
pure_type_count.rename(columns={"type_1":"Type",0:"Count"},inplace=True)

sns.catplot(data=pure_type_count,x="Type",y="Count",kind="bar",aspect=2.5,palette=type_colors.array[1:])
plt.xlabel("Pure Pokemon Types")
plt.ylabel("Count")
plt.title("Distribution of Pure Type Pokemon")


mix_type_count = (pokedex[pokedex["type_2"]!=0].groupby("type_2").apply(lambda df: df.shape[0]) + 
                  pokedex[pokedex["type_2"]!=0].groupby("type_1").apply(lambda df: df.shape[0])).rename(lambda x: type_encoding.index[x]).reset_index()


mix_type_count.rename(columns={"type_2":"Type",0:"Count"},inplace=True)

sns.catplot(data=mix_type_count,x="Type",y="Count",kind="bar",aspect=2.5,palette=type_colors.array[1:])
plt.xlabel("Mixed Pokemon Types")
plt.ylabel("Count")
plt.title("Distribution of Mixed Type Pokemon")

total_count = (pure_type_count["Count"] + mix_type_count["Count"]).rename(lambda x: type_encoding.index[x+1]).reset_index()
total_count.rename(columns={"index":"Type"},inplace=True)

sns.catplot(data=total_count,x="Type",y="Count",kind="bar",aspect=2.5,palette=type_colors.array[1:])
plt.xlabel("Pokemon Types")
plt.ylabel("Count")
plt.title("Total Distribution of Pokemon Types")
```




    Text(0.5, 1.0, 'Total Distribution of Pokemon Types')




    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    



    
![png](output_26_3.png)
    



```python
# Conclusions on Type Distributions:

# Among Pure Types, the most common is the Water-type and a close next comes the Normal-Type. The least frequent is the Flying-type.

# Among Mixed the most common is the Flying-type, far outnumbering any other Type. The least frequent is the Ice-type.

# In all, the most frequent type is the Water-type and the least frequent type is the Ice-type.
```


```python
# Pokemon Battle Stats columns:

bt_sts = ['total_points', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense','speed']

# The properties 'total_points', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed' 
# are very familiar to anyone that has ever played a Pokemon game.

# These *positive integer* values are some of the key descriptions of a Pokemon in Battles:
# 'hp' or Health Points keeps a Pokemon fighting, and when it hits 0 a Pokemon "faints" and loses.
# 'attack' and 'sp_attack' determine the damage that a Pokemon's moves inflict.
# 'defense' and 'sp_defense' determine the damage that is dealt to the Pokemon upon being hit by a move.
# 'speed' determines the move-order of Pokemon in battle. The faster Pokemon attack first.

# 'total_points' is a simple aggregate of all the individual properties mentioned above. Let's verify this:
print("Does any exception exist:",(pokedex["total_points"] != (pokedex.loc[:,bt_sts[1:]].apply(lambda s: s.sum(),axis="columns"))).any())

```

    Does any exception exist: False



```python
# Distribution of parameter values among different types:
# For Pokemon of each Type(in either 1 or 2), let us examine the distribution of the various parameters.

# We can define an "average" Pokemon to form a Baseline for comparison:

# The overall "average" attribute Pokemon : (From previous summary table) 
# total_points: 439.35311
# hp: 70.067943
# attack: 80.476555
# defense: 74.670813
# sp_attack: 73.031579
# sp_defense: 72.298565
# speed: 68.807656

# Moreover, let us also form seperately form the "average" Pokemon of each Type:
avg_poke_stats = pd.DataFrame(0,index=type_encoding.index,columns=bt_sts[1:])
avg_poke_stats.loc["Null",bt_sts[1:]] = pokedex[bt_sts[1:]].apply(lambda s: s.mean(),axis="index")



def plot_stat(pokedex,table,stat,type_colors):
    fig , axes = plt.subplots(4,5,figsize=(25,15))
    axes[3,3].remove()
    axes[3,4].remove()
    for t in range(1,19):
        data = pokedex[(pokedex["type_1"]==t) | (pokedex["type_2"]==t)]
        table.loc[(table.index[t]),stat] = data.loc[:,stat].mean()
        axes[(t-1)//5,(t-1)%5].set_xlim(0,260)
        sns.histplot(data=data,x=stat,color=type_colors[t],ax=axes[(t-1)//5,(t-1)%5],alpha=0.7,binwidth=10)
        axes[(t-1)//5,(t-1)%5].set(xlabel=None,ylabel=None)
    fig.tight_layout()
    plt.show()
```


```python
plot_stat(pokedex,avg_poke_stats,"hp",type_colors) # Health Points
```


    
![png](output_30_0.png)
    



```python
plot_stat(pokedex,avg_poke_stats,"attack",type_colors) # Attack
```


    
![png](output_31_0.png)
    



```python
plot_stat(pokedex,avg_poke_stats,"defense",type_colors) # Defense
```


    
![png](output_32_0.png)
    



```python
plot_stat(pokedex,avg_poke_stats,"sp_attack",type_colors) # Special Attack
```


    
![png](output_33_0.png)
    



```python
plot_stat(pokedex,avg_poke_stats,"sp_defense",type_colors) # Special Defense
```


    
![png](output_34_0.png)
    



```python
plot_stat(pokedex,avg_poke_stats,"speed",type_colors) # Speed
```


    
![png](output_35_0.png)
    



```python
# Comparing the average Pokemon of each Type:
avg_poke_stats.rename(lambda s: ' '.join(str.split(s,sep='_')).upper(),axis="columns",inplace=True)

x = np.arange(6)  # The number of x-axis labels
width = 0.04  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(18,7))

for ind in range(19):
    offset = width * multiplier
    rects = ax.bar(x + offset, avg_poke_stats.iloc[ind], width, label=avg_poke_stats.index[ind],color=type_colors.iloc[ind])
    multiplier += 1
ax.set_xticks(x + width*9, avg_poke_stats.columns)
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(50)
plt.show()

fig,axis = plt.subplots(figsize=(18,7))
axis.bar(range(6,19*6+1,6),avg_poke_stats.apply(lambda s: s.sum(),axis="columns").array,2,color=type_colors)
axis.set_xticks(range(6,19*6+1,6), avg_poke_stats.index)
plt.title("Total Points on Average per Type")
plt.show()

```


    
![png](output_36_0.png)
    



    
![png](output_36_1.png)
    



```python
# Conclusions on columns of Battle Statistics:

# When considering the average values of the properties grouped by Type, we observe:

# Dragon-Types have the highest HP.
# Fighting-Types have the highest Attack.
# Steel-Types and Rock-Types have the highest Defense.
# Dragon-Types and Psychic-Types have the highest Special Attack.
# Psychic-Types followed by Fairy-Types and Dragon-Types have the highest Special Defense.
# Flying-Types followed by Electric-Types and Dragon-Types have the highest Speed.

# On average, Dragon-Types have the highest cumulative Total Points, while the lowest is of Bug-Types.
```


```python
# Pokedex Number - Generation

sns.relplot(data=pokedex,x="pokedex_number",y="generation",hue="status",palette=sns.color_palette("viridis",as_cmap=True),height=5,aspect=2.6)
plt.xlabel("Poxedex Numbers")
plt.ylabel("Generation Tags")
plt.title("Relation of Pokedex Numbers with Generation Tags")
plt.show()

# Exceptional Pokemon from Generation 8:
pokedex[(pokedex["generation"]==8) & (pokedex["pokedex_number"]<200)].loc[:,['pokedex_number','name','generation','status','species','type_1','type_2']]

# Observations:

# Barring the exceptional Pokemon(all of which have the same prefix value 2), the Generation and Pokedex Number 
# seem to share a "step-functional" relationship.

# Moreover, looking at the Distribution of Status, we observe, in each Generation, the Pokedex numbers are ordered 
# in increasing manner of Status tag.

# The only generation without any Status-4 Pokemon: Generation 8.
```


    
![png](output_38_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>104</th>
      <td>79</td>
      <td>Galarian Slowpoke</td>
      <td>8</td>
      <td>1</td>
      <td>44</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>80</td>
      <td>Galarian Slowbro</td>
      <td>8</td>
      <td>1</td>
      <td>45</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>185</th>
      <td>144</td>
      <td>Galarian Articuno</td>
      <td>8</td>
      <td>2</td>
      <td>95</td>
      <td>15</td>
      <td>8</td>
    </tr>
    <tr>
      <th>187</th>
      <td>145</td>
      <td>Galarian Zapdos</td>
      <td>8</td>
      <td>2</td>
      <td>96</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>189</th>
      <td>146</td>
      <td>Galarian Moltres</td>
      <td>8</td>
      <td>2</td>
      <td>97</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>246</th>
      <td>199</td>
      <td>Galarian Slowking</td>
      <td>8</td>
      <td>1</td>
      <td>130</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Base Friendship - Status

sns.catplot(data=pokedex,x="status",y="base_friendship",hue="status",palette=sns.color_palette("viridis",as_cmap=True),height=6,aspect=2,jitter=.3)
plt.xlabel("Status of Pokemon")
plt.ylabel("Base Friendship")
plt.title("Relation between Base Friendship and Status Tag")
```




    Text(0.5, 1.0, 'Relation between Base Friendship and Status Tag')




    
![png](output_39_1.png)
    



```python
# Growth Rate - Status

sns.catplot(data=pokedex,y="status",x="growth_rate",hue="status",palette=sns.color_palette("viridis",as_cmap=True),height=4,aspect=3,jitter=0.35)
plt.ylabel("Status")
plt.xlabel("Growth Rate")
plt.title("Growth rates of Pokemon of different Status")
plt.show()

# Exceptional Status-4 and Status-1 Pokemon:
pokedex[((pokedex["status"]==4) & (pokedex["growth_rate"]==4)) | ((pokedex["status"]==1) & (pokedex["growth_rate"]==0))].loc[:,['pokedex_number','name','generation','status','species','type_1','type_2',"growth_rate"]]

```


    
![png](output_40_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>growth_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>151</td>
      <td>Mew</td>
      <td>1</td>
      <td>4</td>
      <td>99</td>
      <td>15</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>304</th>
      <td>251</td>
      <td>Celebi</td>
      <td>2</td>
      <td>4</td>
      <td>168</td>
      <td>15</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>588</th>
      <td>492</td>
      <td>Shaymin Land Forme</td>
      <td>4</td>
      <td>4</td>
      <td>359</td>
      <td>10</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>589</th>
      <td>492</td>
      <td>Shaymin Sky Forme</td>
      <td>4</td>
      <td>4</td>
      <td>359</td>
      <td>10</td>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>658</th>
      <td>555</td>
      <td>Galarian Darmanitan Zen Mode</td>
      <td>5</td>
      <td>1</td>
      <td>407</td>
      <td>12</td>
      <td>7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Base EXP - Catch Rate

# sns.relplot(data=pokedex.groupby("catch_rate").apply(lambda df: df["total_points"].sum()/df.shape[0]),aspect=2,kind="line")
sns.relplot(data=pokedex,x="catch_rate",y="base_experience",kind="line",aspect=2.5)
plt.xlabel("Catch Rate")
plt.ylabel("Base Experience")
plt.title("Relation between Base Experience and Catch Rate")
```




    Text(0.5, 1.0, 'Relation between Base Experience and Catch Rate')




    
![png](output_41_1.png)
    



```python
# Base EXP - Total points - Catch Rate

sns.relplot(data=pokedex,y="base_experience",x="total_points",hue="catch_rate",palette=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True),height=4,aspect=3)
plt.xlabel("Base Experience Value")
plt.ylabel("Total Points")
plt.title("Relation of Base EXP with Total Points")
```




    Text(0.5, 1.0, 'Relation of Base EXP with Total Points')




    
![png](output_42_1.png)
    



```python
# Base EXP - Total points - Status

sns.relplot(data=pokedex,x="base_experience",y="total_points",hue="status",palette=sns.color_palette("viridis",as_cmap=True),height=5,aspect=2)
plt.xlabel("Base Experience Value")
plt.ylabel("Total Points")
plt.title("Relation of Base EXP with Total Points")
```




    Text(0.5, 1.0, 'Relation of Base EXP with Total Points')




    
![png](output_43_1.png)
    


##### Analysis on Damage Multipliers properties of different Types of Pokemon: (Columns: D1 to D18)


```python
# Initial analysis of 'DM' columns:

# The 'against_?' columns, although of type 'float64', have very 
# few unique values(ranging from 5 to 8). Hence they might as-well
# be considered categorical variables of a Pokemon.

# If one has ever played a Pokemon game, one would know these words:

#   Fire beats Grass. Grass beats Water. Water beats Fire.

# However, does the "real-world" data on Pokemons actually back such claims?
# Is there a defined relation between Damage multipliers and Pokemon Types?

# Furthermore, if there is such a relation, then how does it differ 
# for Pure Type Pokemons and Mixed Type Pokemons?

# Let us systematically examine the data to find such interesting 
# properties of Pokemon, if they even exist.
```


```python
# Let us examine the relation of Type and its corresponding 
# Damage multipliers, seperately for Pure Type and Mixed Type Pokemon.

# For Pure Type Pokemon:
poke_pure = (pokedex.loc[pokedex['type_2']==0]).drop(columns=["type_2"],errors="ignore")
# For Mixed Type Pokemon:
poke_mixed = (pokedex.loc[pokedex['type_2']!=0]).copy()
```


```python
# Defining the Relation that we are in search for:

#     We aim to draw a generalization of the form:
     
#         For all Pokemon known in existence,
#         Any 2 Pokemon that have the same Type, must have the same Damage Multiplier values, 
#         for any specific Type of Damage Multiplier being considered.

# Type being a property ***completely*** defined by the 'type_1' and 'type_2' columns of a Pokemon.

# It would mean, that The Damage Multiplier of any Pokemon against any Type 
# is independent of all variables which are independent of the Type variables of that Pokemon.
# Being able to prove such a statement would greatly reduce the complexity of our analysis.
```


```python
# Finding relations between Type and Damage Multiplliers for Pure Types:

def calc_defs(df):
    s = pd.Series(dtype=object)
    for i in range(1,19): # 18 Types
        s[f"D{i}"] = dict(df[f"D{i}"].value_counts())
    return s

def_stats_pure = poke_pure.groupby("type_1").apply(calc_defs)
def_stats_pure.index = (type_encoding.iloc[def_stats_pure.index]).index
```


```python
# An interesting way to check how well the relation holds is to see 
# for what fraction of table entries, the list of possibilities 
# has a length exactly 1. This is implemented in the following function.

def check_relation_validity(table):
    valid_entries = table.apply(lambda s: s.apply(lambda v: (len(v)==1) | ((min(v.keys())>1)^(max(v.keys())<1))).sum(),axis="index").sum()
    total_entries = table.shape[0]*table.shape[1]
    return valid_entries/total_entries*100

print("For Pure Types the relation holds",check_relation_validity(def_stats_pure),"\x25 of time.")
def_stats_pure.apply(lambda s: s.apply(lambda v:int((len(v)==1) | ((min(v.keys())>1)^(max(v.keys())<1)))),axis="index")

# Notice the anomalies in the relation. Most of the anomalous behaviour seems to be with the Psychic and Ice Type Pokemon. 
```

    For Pure Types the relation holds 91.35802469135803 % of time.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
      <th>D16</th>
      <th>D17</th>
      <th>D18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Hence we observe that, **barring few exceptions**, most Pure Type 
# Pokemon of 1 Type, have the same Damage Multiplier against some type.

# Let us try to study some of these exceptions:

# One exception is in "Dragon" Type Pokemon, in the 'D10' column.
# Let us see how varied the Dragon types are in their defense against Grass(10):
print(def_stats_pure.loc["Dragon","D10"])

# The majority(10) of "Dragon" Types have 0.5 multiplier against Grass(10) Types. 
# There are 3 Pure Dragon Types which are "IMMUNE"(i.e. 0.0 multiplier) to Grass(10) Type:
poke_pure.loc[(poke_pure["type_1"]==type_encoding["Dragon"]) & (poke_pure["D10"]==0),["pokedex_number","generation","status","species"]]


# When speaking in an "approximate"/"aggregate"/"overall" manner, we might be permitted to omit such 
# exceptions to make a generalized statement on the Type dependency of 
# Damage multipliers BY ONLY CONSIDERING THE HIGHEST FREQUENCY VALUE.

#However, such a description will not be 100% accurate but more probabilistic.
```

    {0.5: 10, 0.0: 3}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>819</th>
      <td>704</td>
      <td>6</td>
      <td>1</td>
      <td>518</td>
    </tr>
    <tr>
      <th>820</th>
      <td>705</td>
      <td>6</td>
      <td>1</td>
      <td>518</td>
    </tr>
    <tr>
      <th>821</th>
      <td>706</td>
      <td>6</td>
      <td>1</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mixed Type 'DM' analysis:

# Considering Primary Type only:
def_stats_mix = poke_mixed.groupby("type_1").apply(calc_defs)
# def_stats_mix.reset_index(inplace=True)
# def_stats_mix['type_1'] = def_stats_mix['type_1'].apply(lambda x: type_encoding.index[x])
print("Considering only Mixed Primary Types the relation holds",check_relation_validity(def_stats_mix),"\x25 of time.")
# There does not seem to be any prevalent relation observed.

# Considering Secondary Type only:
def_stats_mix = poke_mixed.groupby("type_2").apply(calc_defs)
# def_stats_mix.reset_index(inplace=True)
# def_stats_mix['type_2'] = def_stats_mix['type_2'].apply(lambda x: type_encoding.index[x])
print("Considering only Mixed Secondary Types the relation holds",check_relation_validity(def_stats_mix),"\x25 of time.")
# There does not seem to be any prevalent relation observed.

# There is scope for improvement in each measure, but not significant.
```

    Considering only Mixed Primary Types the relation holds 7.098765432098765 % of time.
    Considering only Mixed Secondary Types the relation holds 9.5679012345679 % of time.



```python
# Considering both Primary and Secondary Types both:
def_stats_mix = poke_mixed.groupby(["type_1","type_2"]).apply(calc_defs)
print("Considering both Primary and Secondary Types the relation holds",check_relation_validity(def_stats_mix),"\x25 of time.")
# **Barring few exceptions**, a relation is observed!
```

    Considering both Primary and Secondary Types the relation holds 98.78671775223499 % of time.



```python
# Now, does the order of the Types matter? 
# Are Damage multipliers of (Bug, Electric) Type the same as that 
# of a (Electric, Bug) Type? Let us combine them and check.

# Orders the Primary and Secondary types to align types (a,b) and (b,a) for all 'a','b' in 'type_encoding'.
def fix_order(s):
    s.iloc[0],s.iloc[1] = min(s.iloc[0],s.iloc[1]),max(s.iloc[0],s.iloc[1])
    return s

def merge_types(df:pd.DataFrame):
    def merge_dicts(s):
        s_new = {}
        for i in s: # 'i' is a 'dict'
            for j in i: # 'j' is a dictionary 'key' (Cannot be -1)
                if(s_new.get(j,-1)==-1):
                    s_new[j] = i[j]
                else:
                    s_new[j] += i[j]
        return s_new        
    return df.apply(merge_dicts,axis="index")

def_stats_mix.reset_index(inplace=True)
def_stats_mix.loc[:,["type_1","type_2"]] = (def_stats_mix.loc[:,["type_1","type_2"]]).apply(fix_order,axis="columns")
def_stats_mix = def_stats_mix.groupby(['type_1','type_2']).apply(merge_types).reset_index()

def_stats_mix.sort_values(["type_1","type_2"],axis="index",inplace=True)

def_stats_mix["type_1"] = def_stats_mix["type_1"].apply(lambda x: type_encoding.index[x])
def_stats_mix["type_2"] = def_stats_mix["type_2"].apply(lambda x: type_encoding.index[x])

def_stats_mix.set_index(['type_1',"type_2"],inplace=True,drop=True)
def_stats_mix.rename_axis(index=[None,None],inplace=True)

print("For Mixed Types the *unordered relation* holds",check_relation_validity(def_stats_mix),"\x25 of time.")

# We see that the "relation strength" does not drop noticeably.
# Hence, we conclude that the relation between Damage Multiplier and Pokemon Type for Mixed Type Pokemon is independent 
# of the specific ordering of the types, as Primary or Secondary. 

# Thus, the property of Damage Multipliers(as a function of Type) is symmetric about the Primary and Secondary type variables of a Pokemon, to some extent, probably.
def_stats_mix.apply(lambda s: s.apply(lambda v: int((len(v)==1) | ((min(v.keys())>1)^(max(v.keys())<1)))),axis="index")
```

    For Mixed Types the *unordered relation* holds 98.16176470588235 % of time.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
      <th>D16</th>
      <th>D17</th>
      <th>D18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Bug</th>
      <th>Electric</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Psychic</th>
      <th>Steel</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Rock</th>
      <th>Steel</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Steel</th>
      <th>Water</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>136 rows × 18 columns</p>
</div>




```python
# Closing remarks on 'DM' analysis:

# Coming back to the inspiration to our analysis:
#   1) Does Fire beat Grass? 
#   2) Does Grass beat Water? 
#   3) Does Water beat Fire?

# Let us denote this attribute Damage multiplier by the notation: DM(A,B) ; where 'A' is any Pure/Mixed Type and 'B' is any Pure Type. 

# By definition if Damage multipliers are greater than 1 for Type A against Type B, then Type B is more effective against Type A. DM(A,B) > 1 ; (a.k.a B beats A)
# By definition if Damage multipliers are less than 1 for Type A against Type B, then Type B is less effective against Type A. DM(A,B) < 1 ;
# By definition if Damage multipliers are exactly equal to 1 for Type A against Type B, then Type B is unbiased to Type A. DM(A,B) = 1 ;

#An interesting observation would be that although 'Type A' can be either Pure or Mixed, 'Type B' can only be Pure.(According to given data)
# Thus it might be worth asking as to why there are no Mixed types when it comes to the Damage Multiplier columns(of which there are only 18).
```


```python
# Question 1:
# For Pure Types:
print(def_stats_pure.loc["Grass","D7"])
print()
# For Mixed Types:
try:
    print(def_stats_mix.loc[("Grass",slice(None)),"D7"])
except(KeyError):
    pass
try:
    print(def_stats_mix.loc[(slice(None),"Grass"),"D7"])
except(KeyError):
    pass

# For all 43 Pure Types it is True.

# Incase of Mixed Types, it holds True for 
# 65 Mixed Grass-? types except the following:
    
#     1 Poison-Grass type Pokemon
#     All 2 Grass-Rock type Pokemon
#     All 3 Grass-Water type Pokemon
#     All 5 Dragon-Grass type Pokemon

# A total of 11 exceptions.
# Hence, for Mixed Types it holds True, around 85.53% of cases.

```

    {2.0: 43}
    
    Grass  Ground              {2.0: 1}
           Ice                 {4.0: 3}
           Normal              {2.0: 2}
           Poison     {2.0: 14, 1.0: 1}
           Psychic             {2.0: 4}
           Rock                {1.0: 2}
           Steel               {4.0: 3}
           Water               {1.0: 3}
    Name: D7, dtype: object
    Bug       Grass     {4.0: 6}
    Dark      Grass     {2.0: 4}
    Dragon    Grass     {1.0: 5}
    Electric  Grass     {2.0: 1}
    Fairy     Grass     {2.0: 5}
    Fighting  Grass     {2.0: 3}
    Flying    Grass     {2.0: 7}
    Ghost     Grass    {2.0: 12}
    Name: D7, dtype: object



```python
# Question 2:
# For Pure Types:
print(def_stats_pure.loc["Water","D10"])
print()
# For Mixed Types:
try:
    print(def_stats_mix.loc[("Water",slice(None)),"D10"])
except(KeyError):
    pass
try:
    print(def_stats_mix.loc[(slice(None),"Water"),"D10"])
except(KeyError):
    pass


# For all 72 Pure Types it is True.

# Incase of Mixed Types, it holds True for 
# 55 Mixed Water-? types except the following:
    
#     Bug-Water types: 5
#     Dragon-Water types: 3
#     Fire-Water types: 1
#     Flying-Water types: 8
#     Grass-Water types: 3
#     Poison-Water types: 6
#     Steel-Water types: 1

# A total of 27 exceptions.
# Hence, for Mixed Types it holds True, around 67.07% of cases.
```

    {2.0: 72}
    
    Bug       Water            {1.0: 5}
    Dark      Water            {2.0: 7}
    Dragon    Water            {1.0: 3}
    Electric  Water            {2.0: 3}
    Fairy     Water            {2.0: 4}
    Fighting  Water            {2.0: 4}
    Fire      Water            {1.0: 1}
    Flying    Water            {1.0: 8}
    Ghost     Water            {2.0: 2}
    Grass     Water            {1.0: 3}
    Ground    Water           {4.0: 10}
    Ice       Water            {2.0: 7}
    Normal    Water            {2.0: 1}
    Poison    Water            {1.0: 6}
    Psychic   Water            {2.0: 6}
    Rock      Water    {4.0: 9, 3.0: 2}
    Steel     Water            {1.0: 1}
    Name: D10, dtype: object



```python
# Question 3:
# For Pure Types:
print(def_stats_pure.loc["Fire","D18"])
print()
# For Mixed Types:
try:
    print(def_stats_mix.loc[("Fire",slice(None)),"D18"])
except(KeyError):
    pass
try:
    print(def_stats_mix.loc[(slice(None),"Fire"),"D18"])
except(KeyError):
    pass

# For all 34 Pure Types it is True.

# Incase of Mixed Types, it holds True for 
# 44 Mixed Fire-? types except the following:
    
#     Fire-Ground types: 1
#     Fire-Water types: 1
#     Dragon-Fire types: 3

# A total of 5 exceptions.
# Hence, for Mixed Types it holds True, around 89.80% of cases.

```

    {2.0: 34}
    
    Fire  Flying                     {2.0: 7}
          Ghost                      {2.0: 5}
          Ground     {4.0: 2, 3.0: 1, 0.0: 1}
          Ice                        {2.0: 1}
          Normal                     {2.0: 2}
          Poison                     {2.0: 2}
          Psychic                    {2.0: 3}
          Rock                       {4.0: 3}
          Steel                      {2.0: 1}
          Water                      {0.0: 1}
    Name: D18, dtype: object
    Bug       Fire    {2.0: 4}
    Dark      Fire    {2.0: 4}
    Dragon    Fire    {1.0: 3}
    Electric  Fire    {2.0: 1}
    Fighting  Fire    {2.0: 7}
    Name: D18, dtype: object



```python
# Closing remarks on 'DM' analysis:

# When considering the Pure Types only, all statements 
# are always True, for all 149/149 Pure Type Pokemon.
# (Interestingly, out of the 492 Pure Type Pokemon, 149 belong to either Fire, Water, or Grass types)

# When considering Mixed Types: 

#     The first statement holds true 65/76 times.
#     The second statement holds true 55/82 times.
#     The third statement holds true 44/49 times.

# Thus on average, for a Mixed Type, the statements are True 164/207 of times.

# In total, for all Pokemon to which these statements apply, the statements are True 313/356 (87.92%) of times.

# The exceptions are somewhat reasonable and obvious, for there is ambiguity 
# in interpretation of the statements for the following types of Pokemon:

#     Fire-Water Type ; Grass-Water Type ;  (and the non-existent) Fire-Grass Type ;


# Yes, Indeed. On Venturing far, we Pilgrims Discover the Truth of the Old Words:

#     Fire beats Grass. Grass beats Water. Water beats Fire.
```


```python
# Additional analysis of 'DM' columns:

# An interesting follow-up question might be: 
# If Type A is more effective against Type B , then is Type B less effective against Type A ? Does the converse hold True as well?

# Let us interpret the first of the 2 questions by the following equation: (DM(A,B) < 1 <= DM(B,A) > 1) ; (<= is the implication operator in this context)
# Due to the absence of Mixed type columns for Damage multipliers, 
# we can only evidently define such questions when both Type B and Type A are Pure.


# The 'def_stats_pure' is an 18x18 DataFrame, corresponding to the 18 Types, the axes in sorted order of 'type_encoding':
print(def_stats_pure.shape)

# As the rows and columns are in **sorted order** of 'type_encoding', DM(A,B) and DM(B,A) 
# are simply reflections about the "main" diagonal of the square matrix 'def_stats_pure'
def_stats_pure
```

    (18, 18)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
      <th>D16</th>
      <th>D17</th>
      <th>D18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{0.5: 19}</td>
      <td>{2.0: 19}</td>
      <td>{2.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{0.5: 19}</td>
      <td>{0.5: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{2.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>{2.0: 14}</td>
      <td>{0.5: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{2.0: 14}</td>
      <td>{2.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{0.5: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{0.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
      <td>{1.0: 14}</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>{1.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{2.0: 13}</td>
      <td>{0.5: 13}</td>
      <td>{2.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{0.5: 13}</td>
      <td>{1.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{0.5: 10, 0.0: 3}</td>
      <td>{1.0: 13}</td>
      <td>{2.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{1.0: 13}</td>
      <td>{0.5: 13}</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{0.5: 26, 0.0: 8}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{0.5: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{2.0: 31, 0.0: 3}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{0.5: 34}</td>
      <td>{1.0: 34}</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>{0.5: 19}</td>
      <td>{0.5: 19}</td>
      <td>{0.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{0.5: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{2.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{1.0: 19}</td>
      <td>{2.0: 19}</td>
      <td>{1.0: 19}</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>{0.5: 29}</td>
      <td>{0.5: 29}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 29}</td>
      <td>{2.0: 29}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 27, 0.5: 2}</td>
      <td>{2.0: 29}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 27, 0.5: 2}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 29}</td>
      <td>{2.0: 29}</td>
      <td>{0.5: 29}</td>
      <td>{1.0: 29}</td>
      <td>{1.0: 29}</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>{0.5: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{0.5: 34}</td>
      <td>{1.0: 34}</td>
      <td>{0.5: 31, 0.0: 3}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{0.5: 34}</td>
      <td>{2.0: 34}</td>
      <td>{0.5: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{1.0: 34}</td>
      <td>{2.0: 34}</td>
      <td>{0.5: 34}</td>
      <td>{2.0: 34}</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>{0.5: 4}</td>
      <td>{1.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{2.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{0.5: 4}</td>
      <td>{1.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{0.5: 4}</td>
      <td>{0.0: 4}</td>
      <td>{2.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{2.0: 4}</td>
      <td>{1.0: 4}</td>
      <td>{1.0: 4}</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>{0.5: 15}</td>
      <td>{2.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{0.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{2.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 12, 0.0: 3}</td>
      <td>{1.0: 15}</td>
      <td>{0.0: 15}</td>
      <td>{0.5: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 15}</td>
      <td>{1.0: 15}</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>{2.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{0.5: 43}</td>
      <td>{1.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{2.0: 43}</td>
      <td>{2.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{0.5: 41, 0.0: 2}</td>
      <td>{0.5: 42, 0.0: 1}</td>
      <td>{2.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{2.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{1.0: 43}</td>
      <td>{0.5: 42, 0.0: 1}</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>{1.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{0.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{2.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{2.0: 17}</td>
      <td>{1.0: 17}</td>
      <td>{0.5: 17}</td>
      <td>{1.0: 17}</td>
      <td>{0.5: 17}</td>
      <td>{1.0: 17}</td>
      <td>{2.0: 17}</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>{1.0: 19, 0.5: 1}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 19, 0.5: 1}</td>
      <td>{2.0: 19, 1.0: 1}</td>
      <td>{2.0: 19, 0.5: 1}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 19, 0.5: 1}</td>
      <td>{1.0: 18, 2.0: 1, 0.0: 1}</td>
      <td>{0.5: 20}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 20}</td>
      <td>{1.0: 20}</td>
      <td>{2.0: 20}</td>
      <td>{2.0: 19, 0.5: 1}</td>
      <td>{1.0: 19, 2.0: 1}</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{2.0: 71}</td>
      <td>{1.0: 67, 0.5: 2, 2.0: 2}</td>
      <td>{1.0: 71}</td>
      <td>{0.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 69, 0.5: 2}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
      <td>{1.0: 71}</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>{0.5: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{0.5: 16}</td>
      <td>{0.5: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{0.5: 16}</td>
      <td>{2.0: 14, 0.0: 2}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{0.5: 16}</td>
      <td>{2.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>{2.0: 43, 0.5: 1, 1.5: 1}</td>
      <td>{2.0: 43, 1.0: 1, 1.5: 1}</td>
      <td>{1.0: 45}</td>
      <td>{1.0: 45}</td>
      <td>{1.0: 44, 0.5: 1}</td>
      <td>{0.5: 44, 1.0: 1}</td>
      <td>{1.0: 42, 0.5: 2, 0.0: 1}</td>
      <td>{1.0: 45}</td>
      <td>{2.0: 43, 1.0: 1, 1.5: 1}</td>
      <td>{1.0: 44, 0.5: 1}</td>
      <td>{1.0: 37, 0.0: 7, 2.0: 1}</td>
      <td>{1.0: 42, 0.5: 3}</td>
      <td>{1.0: 45}</td>
      <td>{1.0: 45}</td>
      <td>{0.5: 44, 1.0: 1}</td>
      <td>{1.0: 44, 2.0: 1}</td>
      <td>{1.0: 44, 0.5: 1}</td>
      <td>{1.0: 44, 2.0: 1}</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{2.0: 16}</td>
      <td>{0.5: 16}</td>
      <td>{0.5: 16}</td>
      <td>{1.0: 16}</td>
      <td>{2.0: 16}</td>
      <td>{2.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{0.5: 16}</td>
      <td>{0.5: 16}</td>
      <td>{1.0: 16}</td>
      <td>{1.0: 16}</td>
      <td>{2.0: 16}</td>
      <td>{2.0: 16}</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>{0.5: 11}</td>
      <td>{1.0: 11}</td>
      <td>{0.5: 11}</td>
      <td>{1.0: 11}</td>
      <td>{0.5: 11}</td>
      <td>{2.0: 10, 1.5: 1}</td>
      <td>{2.0: 10, 1.5: 1}</td>
      <td>{0.5: 11}</td>
      <td>{1.0: 11}</td>
      <td>{0.5: 11}</td>
      <td>{2.0: 10, 1.5: 1}</td>
      <td>{0.5: 11}</td>
      <td>{0.5: 11}</td>
      <td>{0.0: 11}</td>
      <td>{0.5: 11}</td>
      <td>{0.5: 11}</td>
      <td>{0.5: 11}</td>
      <td>{1.0: 11}</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{2.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{0.5: 70, 0.25: 1, 0.0: 1}</td>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{2.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{0.5: 71, 0.25: 1}</td>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{1.0: 72}</td>
      <td>{0.5: 72}</td>
      <td>{0.5: 68, 0.0: 4}</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We saw previously, that most of the anomalies in our relation were due to *Psychic* and *Ice* Type Pokemon.

# Looking at the anomalous entries, we observe that such a behaviour is due to a minority of Pokemon of each such Type.
# For example, (Psychic,D1) has 44 Pokemon with DM > 1 and only 1 Pokemon with DM < 1.
# For example, (Ice,D7) has 19 Pokemon with DM > 1 and only 1 Pokemon with DM < 1.

# Let us modify the table, to only hold the MOST FREQUENT value, before we check for our relation. 
# (This is not an accurate method when there are a large number of exceptions)
def most_freq(d:dict):
    max = -1
    for i in d: # 'd' is a dict
        if(d[i]>d.get(max,-1)):
            max = i
    return max

def show_relation(table:pd.DataFrame):
    if(table.shape[0]!=table.shape[1]):
        return pd.DataFrame(0,index=table.index,columns=table.columns)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            table.iloc[i,j] = ((table.iloc[i,j]==1)*(2*(table.iloc[j,i]==1)-1)) # Relation (3) being tested currently in expression.
    return table

# '-1' means relation is violated, '1' means relation holds. '0' signifies relation does not apply.(i.e. When 'A' is False in the relation A => B)
show_relation(def_stats_pure.apply(lambda s: s.apply(most_freq),axis="index")).astype("int8")

# Primary relation: (1)
# For the relation  (DM(A,B) > 1 => DM(B,A) < 1), we get the following results:
# The relation holds 39/51 times and is violated 12/51 times, of all 51 times that it is tested.

# Converse relation: (2)
# For the converse relation (DM(A,B) < 1 => DM(B,A) > 1), we get the following results:
# The relation holds 14/69 times and is violated 55/69 times, that it is tested.

# Third relation: (3)
# For the relation (DM(A,B) = 1 => DM(B,A) = 1), we get the following results: 
# (We can prove that this relation is equivalent to : (DM(A,B) = 1 <=> DM(B,A) = 1))
# The relation holds 159/204 times and is violated 45/204 times that it is tested.

# Notice that, (51+69+204) = (18*18) = 324 . This is because, for all entries(x) in DM table, either x=1 or x>1 or x<1.
# The final results, even after artificial removal of anomalies, do not hold True in general.

# The cumulative validity comes to around 45.10%. Thus it is approximately as likely to not be True as it is likely to be True.

# NOTE:
# These calculations also do not take into account the weightage that might be 
# given to each of the total number of cases with respect to 
# the NUMBER OF POKEMON, for which it holds or is violated. 
# Uniform distribution of Pokemon among Types and hence equal weightage is assumed.(which is incorrect)

# Some more questions we leave unanswered but might be worth asking:

# Given some subset of Damage Multipliers of a Pokemon, would we be able to deduce the Pokemon's Type? Can we tell if it is Mixed or Pure?
# Do there exist Types about which the Damage Multiplier tables are symmetric?
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
      <th>D16</th>
      <th>D17</th>
      <th>D18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bug</th>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dark</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dragon</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Electric</th>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Fairy</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Fire</th>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Flying</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Ghost</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Grass</th>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ground</th>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ice</th>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Poison</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Psychic</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Steel</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>Water</th>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Conclusions on 'DM' analysis:

#     We deduced that for Mixed and Pure Type Pokemon, there is a strong relation between their Type, 
#     and the effectiveness/ineffectiveness described by Damage Multiplers when facing against other Types.
#     As long as we know a Pokemon's Type, we can, with high level of confidence, predict whether its 
#     corresponding DM values are greater than, less than or equal to 1 against any Type.
pokedex.apply(desc_col,axis="index").T.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Count</th>
      <th>Missing</th>
      <th>Unique Count</th>
      <th>Data Type</th>
      <th>Mean</th>
      <th>Deviation</th>
      <th>Minimum</th>
      <th>Maximum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pokedex_number</td>
      <td>1045</td>
      <td>0</td>
      <td>898</td>
      <td>int64</td>
      <td>440.769378</td>
      <td>262.517231</td>
      <td>1</td>
      <td>898</td>
    </tr>
    <tr>
      <th>1</th>
      <td>name</td>
      <td>1045</td>
      <td>0</td>
      <td>1045</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>generation</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>int64</td>
      <td>4.098565</td>
      <td>2.272788</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>status</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>int64</td>
      <td>1.219139</td>
      <td>0.651271</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>species</td>
      <td>1045</td>
      <td>0</td>
      <td>652</td>
      <td>int64</td>
      <td>285.697608</td>
      <td>198.263458</td>
      <td>0</td>
      <td>651</td>
    </tr>
    <tr>
      <th>5</th>
      <td>type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>18</td>
      <td>int64</td>
      <td>10.407656</td>
      <td>5.524097</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>type_2</td>
      <td>1045</td>
      <td>0</td>
      <td>19</td>
      <td>int64</td>
      <td>5.041148</td>
      <td>5.827109</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>height_m</td>
      <td>1045</td>
      <td>0</td>
      <td>61</td>
      <td>float64</td>
      <td>1.374067</td>
      <td>3.353349</td>
      <td>0.1</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>weight_kg</td>
      <td>1044</td>
      <td>1</td>
      <td>474</td>
      <td>float64</td>
      <td>71.216571</td>
      <td>132.259911</td>
      <td>0.1</td>
      <td>999.9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ability_1</td>
      <td>1045</td>
      <td>0</td>
      <td>213</td>
      <td>int64</td>
      <td>137.22201</td>
      <td>74.445257</td>
      <td>0</td>
      <td>265</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ability_2</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>70.616268</td>
      <td>87.593765</td>
      <td>0</td>
      <td>262</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ability_hidden</td>
      <td>1045</td>
      <td>0</td>
      <td>155</td>
      <td>int64</td>
      <td>111.911005</td>
      <td>89.341561</td>
      <td>0</td>
      <td>266</td>
    </tr>
    <tr>
      <th>12</th>
      <td>total_points</td>
      <td>1045</td>
      <td>0</td>
      <td>216</td>
      <td>int64</td>
      <td>439.35311</td>
      <td>121.992897</td>
      <td>175</td>
      <td>1125</td>
    </tr>
    <tr>
      <th>13</th>
      <td>hp</td>
      <td>1045</td>
      <td>0</td>
      <td>103</td>
      <td>int64</td>
      <td>70.067943</td>
      <td>26.671411</td>
      <td>1</td>
      <td>255</td>
    </tr>
    <tr>
      <th>14</th>
      <td>attack</td>
      <td>1045</td>
      <td>0</td>
      <td>123</td>
      <td>int64</td>
      <td>80.476555</td>
      <td>32.432728</td>
      <td>5</td>
      <td>190</td>
    </tr>
    <tr>
      <th>15</th>
      <td>defense</td>
      <td>1045</td>
      <td>0</td>
      <td>114</td>
      <td>int64</td>
      <td>74.670813</td>
      <td>31.259462</td>
      <td>5</td>
      <td>250</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sp_attack</td>
      <td>1045</td>
      <td>0</td>
      <td>119</td>
      <td>int64</td>
      <td>73.031579</td>
      <td>32.745857</td>
      <td>10</td>
      <td>194</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sp_defense</td>
      <td>1045</td>
      <td>0</td>
      <td>107</td>
      <td>int64</td>
      <td>72.298565</td>
      <td>28.098943</td>
      <td>20</td>
      <td>250</td>
    </tr>
    <tr>
      <th>18</th>
      <td>speed</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>68.807656</td>
      <td>30.210094</td>
      <td>5</td>
      <td>200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>catch_rate</td>
      <td>1045</td>
      <td>0</td>
      <td>36</td>
      <td>float64</td>
      <td>91.141627</td>
      <td>76.183957</td>
      <td>0.0</td>
      <td>255.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>base_friendship</td>
      <td>930</td>
      <td>115</td>
      <td>7</td>
      <td>float64</td>
      <td>64.064516</td>
      <td>21.452532</td>
      <td>0.0</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>base_experience</td>
      <td>925</td>
      <td>120</td>
      <td>180</td>
      <td>float64</td>
      <td>153.716757</td>
      <td>79.28397</td>
      <td>36.0</td>
      <td>608.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>growth_rate</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>int64</td>
      <td>4.196172</td>
      <td>1.084895</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>23</th>
      <td>egg_type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>16</td>
      <td>int64</td>
      <td>7.164593</td>
      <td>3.799033</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>24</th>
      <td>egg_type_2</td>
      <td>1045</td>
      <td>0</td>
      <td>12</td>
      <td>int64</td>
      <td>2.873684</td>
      <td>4.926989</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>25</th>
      <td>percentage_male</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>45.789474</td>
      <td>27.631585</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>egg_cycles</td>
      <td>1045</td>
      <td>0</td>
      <td>11</td>
      <td>float64</td>
      <td>31.479392</td>
      <td>30.49825</td>
      <td>5.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>D13</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.86866</td>
      <td>0.286863</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>D7</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>float64</td>
      <td>1.126316</td>
      <td>0.714569</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>D18</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.050718</td>
      <td>0.609383</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>D4</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.035646</td>
      <td>0.643535</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>D10</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.001196</td>
      <td>0.74435</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>D12</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.675598</td>
      <td>7.685308</td>
      <td>0.0</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>D6</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.07512</td>
      <td>0.753649</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>D14</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.95311</td>
      <td>0.541238</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>D11</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.082297</td>
      <td>0.782683</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>D8</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.1689</td>
      <td>0.592145</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>D15</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>0.977273</td>
      <td>0.501934</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>D1</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.998086</td>
      <td>0.610411</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>D16</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.238278</td>
      <td>0.69656</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>D9</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.01866</td>
      <td>0.568056</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D3</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.977033</td>
      <td>0.375812</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>D2</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.071053</td>
      <td>0.465178</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>D17</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.981579</td>
      <td>0.501753</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>D5</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>1.091148</td>
      <td>0.536285</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Feature Engineering and Encoding 'name' of Pokemon:


```python
# The 'name' of a Pokemon:

"""
The first question that comes to mind: How do we interpret this 'name' property of a Pokemon?

Inspired by our intuition we might interpret it by the name that the 'name' property was given, that being 'name'.

Intuitively such names may not seem to hold any "useful" information about a Pokemon, in the same way that 
most names, such as those given to people, are assumed to be to a great extent arbitrary in comparison 
with other properties of a person. 
Consider this: 

    If I told you that the reason for me being an Electrical Engineer 
    is that my parents named me 'Sounak' when I was born, would you believe me?

However, when considering names attributed in more well developed and scientific fields such as Biology, 
suddenly a name becomes very significant in regards to the meaning it holds and information it gives us about the entity to which 
the name is attributed. Often, such names are given a fancier name, such as Nomenclature or Terminology.

Let us investigate these 'names' and try to form a better understanding of the properties that it might possess.

Let us assume, that 'name's are given in some specific Language(which is simply a fancier word for a set of rules of interpretation) 
and taking hints from the 'german_name' and 'japanese_name' properties, try to find the Language that 'name's hold their meaning in.
"""

# THE 'name' ALPHABET : 

name_ALPH = pd.Series(dtype=np.int64)
for i in pokedex["name"]:
    for j in i:
        name_ALPH[j] = name_ALPH.get(j,0) + 1

name_ALPH.sort_index(inplace=True)
print(name_ALPH.index)
# name_ALPH plot to visualise distributrion.

# Notice, the English alphabet occupying indices '10' to '-4'. Whitespaces ' ' at index '1', may be ommitted in the count.
print("The 'name' alphabet matches the English alphabet with",name_ALPH.iloc[10:-3].sum()/name_ALPH.iloc[:].sum()*100,"\x25 accuracy.")
```

    Index([' ', '%', ''', '-', '.', '0', '1', '2', '5', ':', 'A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
           'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z', 'é', '♀', '♂'],
          dtype='object')
    The 'name' alphabet matches the English alphabet with 96.84199408534009 % accuracy.



```python
# The 'name' of a Pokemon:

"""
Studying this alphabet, might help us understand which language do names come from. However, the method in which 
such a deduction is made, and the foundational knowledge on which it is based is undefined and unclear. What if it 
is a different language from any language ever seen altogether?

Let us call upon our good friend intution to help us out. Intuition says, its English. Well said intuition, well said.

And so, we have made tremendous progress in understanding the 'name' property, now that we know what rules should be 
applied in interpreting a 'name'. Hurray! But how well-defined are the rules of interpretation of English themselves, 
that we use them as the foundational basis of our knowledge of 'name's. It is a natural language after all. ANYTHING GOES!

On the contrary, Mathematics speaks in "well-defined" terms. However, I sometimes 
do wonder, how "well-defined" is the defintion of "well-defined"? WAY TOO SELF REFERENTIAL!!!

In reality, I do believe, that nothing is absolute, its all relative.(SO Nothing IS ABSOLUTE!) So is the case of definitions and 
the algebraic nature present in them along with everything else.

In English, the elementary units of meaning are words. Groups of letters seperated by spaces. It is because I say so.
However, meaning is more convoluted than simply that which is described by words. Often, a specific sequence/ordering of 
words convey a message which is lost/scattered when the meanings of the words are considered seperately in an unordered form.

Thus in English, meaning is contained in not only words, but also specific order of words, called sentences.

Meaning of words given through ordering of letters. Meaning of sentences given through ordering of words. If meaning is all we care about,
Are letters words? Are sentences words? Are letters sentences? Are words sentences? Are words letters? Are sentences letters? If not then why?

However, for the simplicity of our analysis, by the Supreme power vested in me, I claim that any information that 
is contained in such orderings are "useless"/"irrelevant"/"worthless"/"meaningless"/"garbage"/"L" properties of a 'name' 
when it comes to the role they play in describing/defining a Pokemon. 
A "Pokemon", being subject to my definition of it as well! HAAHAHAHAHA! I AM THE RULER HERE!!! WHATEVER I SAY GOES!!!

On the contrary, words are "W"/"interesting"/"significant" properties that a 'name' possesses, and play an important role in 
the description of a Pokemon. Let us explore these words seperated by the ' ' character, in the English language. Why ' '?
"""

# The 'name' LEXICON :

name_LXCO = pd.Series(dtype=np.int64)
for i in pokedex["name"]:
    for j in i.split():
        name_LXCO[j] = name_LXCO.get(j,0) + 1
name_LXCO.sort_values(inplace=True,ascending=False)
print(name_LXCO.head(25))
```

    Mega          48
    Forme         23
    Galarian      20
    Alolan        18
    Form          14
    Size           8
    Mode           6
    Style          6
    Rotom          6
    Tapu           4
    Necrozma       4
    Castform       4
    Gourgeist      4
    Hoopa          4
    Darmanitan     4
    Pumpkaboo      4
    Oricorio       4
    Deoxys         4
    Calyrex        3
    Lycanroc       3
    Mr.            3
    Mewtwo         3
    Zygarde        3
    Wormadam       3
    Incarnate      3
    dtype: int64



```python
# The 'name' of a Pokemon:

""" 
The following observations are made:

"Mega", "Galarian", "Alolan" act as Prefixes to many otherwise common Pokemon names.

"Form", "Forme", "Size", "Mode", "Style" are **parts** of Suffixes to many common Pokemon names. Usually, 
these suffixes are present along with some additional key-word and when seperated along with those key-words, 
multiple Pokemon share the same name.

Some rare suffixes such as 'X' and 'Y' are also found, which are shared by Pokemon, which otherwise have a common name.

Some rare prefixes are also found, such as 'Heat', 'Wash', 'Frost', 'Fan', shared by Pokemon, which otherwise 
have a common name.(The name being 'Rotom')

For the rare prefixes, how did I know that 'Rotom' was not the suffix and these were the prefixes? For the rare suffixes?

HYPOTHESIS:

    Any Pokemon 'name' can be broken down into a 'Suffix', 'Prefix' and a **1-word** name. These **1-word** 
    names all correspond to "similar" Pokemon, while the suffixes and prefixes signal some form of modifications.

    The definition of which Pokemon are to be considered "similar" is still ambiguous. This is because any string 
    of words can be broken down, in multiple ways, into 'Suffix', 'Prefix' and a 1-word residue.

ANOMALIES: Hoopa Hoopa, Mr. Mime ; Mr. Rime ; Mime Jr. ; Type: Null ; (Not 1 word common names)

In our analysis, we start from the most frequently occuring words to be taken as 'suffix'/'prefix' and then proceed to 
lesser frequent words to be considered as suffixes or prefixes.
"""

prefix_encoding = pd.Series(dtype=np.int64)
prefix_encoding[""] = 0
# Create a 'Prefix' column:
pokedex["prefix"] = 0

suffix_encoding = pd.Series(dtype=np.int64)
suffix_encoding[""] = 0
# Creating 'Suffix' column:
pokedex["suffix"] = 0
```


```python
pokedex[pokedex["name"].str.contains("Rotom | Rotom")]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>height_m</th>
      <th>weight_kg</th>
      <th>ability_1</th>
      <th>...</th>
      <th>D15</th>
      <th>D1</th>
      <th>D16</th>
      <th>D9</th>
      <th>D3</th>
      <th>D2</th>
      <th>D17</th>
      <th>D5</th>
      <th>prefix</th>
      <th>suffix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>570</th>
      <td>479</td>
      <td>Heat Rotom</td>
      <td>4</td>
      <td>1</td>
      <td>347</td>
      <td>4</td>
      <td>7</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>109</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>571</th>
      <td>479</td>
      <td>Wash Rotom</td>
      <td>4</td>
      <td>1</td>
      <td>347</td>
      <td>4</td>
      <td>18</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>109</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>572</th>
      <td>479</td>
      <td>Frost Rotom</td>
      <td>4</td>
      <td>1</td>
      <td>347</td>
      <td>4</td>
      <td>12</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>109</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>573</th>
      <td>479</td>
      <td>Fan Rotom</td>
      <td>4</td>
      <td>1</td>
      <td>347</td>
      <td>4</td>
      <td>8</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>109</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>574</th>
      <td>479</td>
      <td>Mow Rotom</td>
      <td>4</td>
      <td>1</td>
      <td>347</td>
      <td>4</td>
      <td>10</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>109</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>




```python
# Encoding Prefixes:

# Encoding the 'Mega', 'Galarian' and 'Alolan' prefixes:
prefix_encoding["Mega"] = prefix_encoding.get("Mega",prefix_encoding.size)
prefix_encoding["Galarian"] = prefix_encoding.get("Galarian",prefix_encoding.size)
prefix_encoding["Alolan"] = prefix_encoding.get("Alolan",prefix_encoding.size)

def set_prefix(s,table,words=1):
    parts = re.split('\s',s["name"],1) # 1 word prefix seperated.

    # t = parts[1] if len(parts)!=1 else ""
    # while(t!="" and words>1):
    #     temp = re.split('\s',t,1)
    #     parts[0] += ' '+temp[0]
    #     parts[1] = temp[1]
    #     t = temp[1] if len(temp!=1) else ""
    #     words -= 1

    if(table.get(parts[0],-1)!=-1):
        s["prefix"] = table[parts[0]]
        if(len(parts)!=1):
            s["name"] = parts[1]
        else:
            s["name"] = ""
    return s

# Update the prefixes and modify the names:
pokedex = pokedex.apply(set_prefix,axis="columns",table=prefix_encoding)

# Encoding Heat, Wash, Frost, Fan and Mow prefixes:
prefix_encoding["Heat"] = prefix_encoding.get("Heat",prefix_encoding.size)
prefix_encoding["Wash"] = prefix_encoding.get("Wash",prefix_encoding.size)
prefix_encoding["Frost"] = prefix_encoding.get("Frost",prefix_encoding.size)
prefix_encoding["Fan"] = prefix_encoding.get("Fan",prefix_encoding.size)
prefix_encoding["Mow"] = prefix_encoding.get("Mow",prefix_encoding.size)

# Update the prefixes and modify the names:
pokedex = pokedex.apply(set_prefix,axis="columns",table=prefix_encoding)

```


```python
pokedex[pokedex["name"].str.contains(" Style|Style ")]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>height_m</th>
      <th>weight_kg</th>
      <th>ability_1</th>
      <th>...</th>
      <th>D15</th>
      <th>D1</th>
      <th>D16</th>
      <th>D9</th>
      <th>D3</th>
      <th>D2</th>
      <th>D17</th>
      <th>D5</th>
      <th>prefix</th>
      <th>suffix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>866</th>
      <td>741</td>
      <td>Oricorio Baile Style</td>
      <td>7</td>
      <td>1</td>
      <td>76</td>
      <td>7</td>
      <td>8</td>
      <td>0.6</td>
      <td>3.4</td>
      <td>38</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>867</th>
      <td>741</td>
      <td>Oricorio Pom-Pom Style</td>
      <td>7</td>
      <td>1</td>
      <td>76</td>
      <td>4</td>
      <td>8</td>
      <td>0.6</td>
      <td>3.4</td>
      <td>38</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>868</th>
      <td>741</td>
      <td>Oricorio Pa'u Style</td>
      <td>7</td>
      <td>1</td>
      <td>76</td>
      <td>15</td>
      <td>8</td>
      <td>0.6</td>
      <td>3.4</td>
      <td>38</td>
      <td>...</td>
      <td>0.5</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>869</th>
      <td>741</td>
      <td>Oricorio Sensu Style</td>
      <td>7</td>
      <td>1</td>
      <td>76</td>
      <td>9</td>
      <td>8</td>
      <td>0.6</td>
      <td>3.4</td>
      <td>38</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>892</td>
      <td>Urshifu Single Strike Style</td>
      <td>8</td>
      <td>2</td>
      <td>644</td>
      <td>6</td>
      <td>2</td>
      <td>1.9</td>
      <td>105.0</td>
      <td>252</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>892</td>
      <td>Urshifu Rapid Strike Style</td>
      <td>8</td>
      <td>2</td>
      <td>644</td>
      <td>6</td>
      <td>18</td>
      <td>1.9</td>
      <td>105.0</td>
      <td>252</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 47 columns</p>
</div>




```python
# Encoding Suffixes:

def set_suffix(s:str,table):
    suf =  re.split('\s',s,1)[1] # First word is the 1-word common name. Rest is the suffix.
    table[suf] = table.get(suf,table.size)
    return table[suf]

# Observation:
# All 'Forme' suffixes are 2 word. All common parts in the names are 1 word.
pokedex.loc[pokedex["name"].str.contains(" Forme"),"suffix"] = pokedex[pokedex["name"].str.contains(" Forme")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Forme"),"name"] = pokedex[pokedex["name"].str.contains(" Forme")]["name"].apply(lambda s: re.split('\s',s,1)[0])


# Observation:
# All 'Form' suffixes are 2 word as well. All common parts in the names are 1 word as well.
pokedex.loc[pokedex["name"].str.contains(" Form"),"suffix"] = pokedex[pokedex["name"].str.contains(" Form")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Form"),"name"] = pokedex[pokedex["name"].str.contains(" Form")]["name"].apply(lambda s: re.split('\s',s,1)[0])


# Observation:
# All 'Size' suffixes are 2 word as well. All common parts in the names are 1 word as well.
pokedex.loc[pokedex["name"].str.contains(" Size"),"suffix"] = pokedex[pokedex["name"].str.contains(" Size")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Size"),"name"] = pokedex[pokedex["name"].str.contains(" Size")]["name"].apply(lambda s: re.split('\s',s,1)[0])


# Observation:
# All but 1 'Mode' suffixes are 2 word as well. All common parts in the names are 1 word as well.
pokedex.loc[pokedex["name"].str.contains(" Mode"),"suffix"] = pokedex[pokedex["name"].str.contains(" Mode")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Mode"),"name"] = pokedex[pokedex["name"].str.contains(" Mode")]["name"].apply(lambda s: re.split('\s',s,1)[0])


# Observation:
# All 'Style' suffixes have 1 word in the common part of their names.
pokedex.loc[pokedex["name"].str.contains(" Style"),"suffix"] = pokedex[pokedex["name"].str.contains(" Style")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Style"),"name"] = pokedex[pokedex["name"].str.contains(" Style")]["name"].apply(lambda s: re.split('\s',s,1)[0])

```


```python
# A different approach:

# Having encoded a lot of suffixes and prefixes, the entires in the 'name_LXCO' table have started 
# to lead us to 1 word Pokemon names. Hence let's change our approach and find multi-word names:
pokedex[pokedex["name"].apply(lambda s: len(s.split()))>1].iloc[:,[0,1,4]]

# To understand whether these Pokemon names contain prefixes/suffixes in them, we can look at the 'pokedex_number' 
# column for hints. If two differently named Pokemon, correspond to the same 'pokedex_number', then we 
# may be correct in our judgement to consider them to be "similar" with some modifications.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>Charizard X</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>Charizard Y</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>25</td>
      <td>Partner Pikachu</td>
      <td>13</td>
    </tr>
    <tr>
      <th>157</th>
      <td>122</td>
      <td>Mr. Mime</td>
      <td>75</td>
    </tr>
    <tr>
      <th>158</th>
      <td>122</td>
      <td>Mr. Mime</td>
      <td>76</td>
    </tr>
    <tr>
      <th>172</th>
      <td>133</td>
      <td>Partner Eevee</td>
      <td>87</td>
    </tr>
    <tr>
      <th>194</th>
      <td>150</td>
      <td>Mewtwo X</td>
      <td>98</td>
    </tr>
    <tr>
      <th>195</th>
      <td>150</td>
      <td>Mewtwo Y</td>
      <td>98</td>
    </tr>
    <tr>
      <th>460</th>
      <td>382</td>
      <td>Primal Kyogre</td>
      <td>274</td>
    </tr>
    <tr>
      <th>462</th>
      <td>383</td>
      <td>Primal Groudon</td>
      <td>275</td>
    </tr>
    <tr>
      <th>496</th>
      <td>413</td>
      <td>Wormadam Plant Cloak</td>
      <td>135</td>
    </tr>
    <tr>
      <th>497</th>
      <td>413</td>
      <td>Wormadam Sandy Cloak</td>
      <td>135</td>
    </tr>
    <tr>
      <th>498</th>
      <td>413</td>
      <td>Wormadam Trash Cloak</td>
      <td>135</td>
    </tr>
    <tr>
      <th>525</th>
      <td>439</td>
      <td>Mime Jr.</td>
      <td>316</td>
    </tr>
    <tr>
      <th>755</th>
      <td>646</td>
      <td>Black Kyurem</td>
      <td>477</td>
    </tr>
    <tr>
      <th>756</th>
      <td>646</td>
      <td>White Kyurem</td>
      <td>477</td>
    </tr>
    <tr>
      <th>791</th>
      <td>678</td>
      <td>Meowstic Male</td>
      <td>496</td>
    </tr>
    <tr>
      <th>792</th>
      <td>678</td>
      <td>Meowstic Female</td>
      <td>496</td>
    </tr>
    <tr>
      <th>844</th>
      <td>720</td>
      <td>Hoopa Hoopa Confined</td>
      <td>528</td>
    </tr>
    <tr>
      <th>845</th>
      <td>720</td>
      <td>Hoopa Hoopa Unbound</td>
      <td>529</td>
    </tr>
    <tr>
      <th>873</th>
      <td>744</td>
      <td>Own Tempo Rockruff</td>
      <td>33</td>
    </tr>
    <tr>
      <th>904</th>
      <td>772</td>
      <td>Type: Null</td>
      <td>568</td>
    </tr>
    <tr>
      <th>918</th>
      <td>785</td>
      <td>Tapu Koko</td>
      <td>577</td>
    </tr>
    <tr>
      <th>919</th>
      <td>786</td>
      <td>Tapu Lele</td>
      <td>577</td>
    </tr>
    <tr>
      <th>920</th>
      <td>787</td>
      <td>Tapu Bulu</td>
      <td>577</td>
    </tr>
    <tr>
      <th>921</th>
      <td>788</td>
      <td>Tapu Fini</td>
      <td>577</td>
    </tr>
    <tr>
      <th>934</th>
      <td>800</td>
      <td>Dusk Mane Necrozma</td>
      <td>589</td>
    </tr>
    <tr>
      <th>935</th>
      <td>800</td>
      <td>Dawn Wings Necrozma</td>
      <td>589</td>
    </tr>
    <tr>
      <th>936</th>
      <td>800</td>
      <td>Ultra Necrozma</td>
      <td>589</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>866</td>
      <td>Mr. Rime</td>
      <td>629</td>
    </tr>
    <tr>
      <th>1012</th>
      <td>875</td>
      <td>Eiscue Ice Face</td>
      <td>283</td>
    </tr>
    <tr>
      <th>1013</th>
      <td>875</td>
      <td>Eiscue Noice Face</td>
      <td>283</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>876</td>
      <td>Indeedee Male</td>
      <td>190</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>876</td>
      <td>Indeedee Female</td>
      <td>190</td>
    </tr>
    <tr>
      <th>1028</th>
      <td>888</td>
      <td>Zacian Crowned Sword</td>
      <td>642</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>888</td>
      <td>Zacian Hero of Many Battles</td>
      <td>642</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>889</td>
      <td>Zamazenta Crowned Shield</td>
      <td>642</td>
    </tr>
    <tr>
      <th>1031</th>
      <td>889</td>
      <td>Zamazenta Hero of Many Battles</td>
      <td>642</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>890</td>
      <td>Eternatus Eternamax</td>
      <td>643</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>898</td>
      <td>Calyrex Ice Rider</td>
      <td>651</td>
    </tr>
    <tr>
      <th>1044</th>
      <td>898</td>
      <td>Calyrex Shadow Rider</td>
      <td>651</td>
    </tr>
  </tbody>
</table>
</div>




```python
pokedex[pokedex["pokedex_number"]==890]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pokedex_number</th>
      <th>name</th>
      <th>generation</th>
      <th>status</th>
      <th>species</th>
      <th>type_1</th>
      <th>type_2</th>
      <th>height_m</th>
      <th>weight_kg</th>
      <th>ability_1</th>
      <th>...</th>
      <th>D15</th>
      <th>D1</th>
      <th>D16</th>
      <th>D9</th>
      <th>D3</th>
      <th>D2</th>
      <th>D17</th>
      <th>D5</th>
      <th>prefix</th>
      <th>suffix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1032</th>
      <td>890</td>
      <td>Eternatus</td>
      <td>8</td>
      <td>3</td>
      <td>643</td>
      <td>14</td>
      <td>3</td>
      <td>20.0</td>
      <td>950.0</td>
      <td>159</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>890</td>
      <td>Eternatus Eternamax</td>
      <td>8</td>
      <td>3</td>
      <td>643</td>
      <td>14</td>
      <td>3</td>
      <td>100.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>...</td>
      <td>2.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 47 columns</p>
</div>




```python
# Encoding remaining Suffixes:

# suffix 'X' and 'Y': All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" X"),"suffix"] = pokedex[pokedex["name"].str.contains(" X")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" X"),"name"] = pokedex[pokedex["name"].str.contains(" X")]["name"].apply(lambda s: re.split('\s',s,1)[0])

pokedex.loc[pokedex["name"].str.contains(" Y"),"suffix"] = pokedex[pokedex["name"].str.contains(" Y")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Y"),"name"] = pokedex[pokedex["name"].str.contains(" Y")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# 'Cloak' suffix: All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" Cloak"),"suffix"] = pokedex[pokedex["name"].str.contains(" Cloak")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Cloak"),"name"] = pokedex[pokedex["name"].str.contains(" Cloak")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# 'Male' and 'Female' suffixes: All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" Male"),"suffix"] = pokedex[pokedex["name"].str.contains(" Male")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Male"),"name"] = pokedex[pokedex["name"].str.contains(" Male")]["name"].apply(lambda s: re.split('\s',s,1)[0])

pokedex.loc[pokedex["name"].str.contains(" Female"),"suffix"] = pokedex[pokedex["name"].str.contains(" Female")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Female"),"name"] = pokedex[pokedex["name"].str.contains(" Female")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# 'Rider' suffix: All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" Rider"),"suffix"] = pokedex[pokedex["name"].str.contains(" Rider")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Rider"),"name"] = pokedex[pokedex["name"].str.contains(" Rider")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# 'Face' suffix: All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" Face"),"suffix"] = pokedex[pokedex["name"].str.contains(" Face")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Face"),"name"] = pokedex[pokedex["name"].str.contains(" Face")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# '... Many Battles', 'Crowned Shield' and 'Crowned Sword' suffixes: All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" Battles"),"suffix"] = pokedex[pokedex["name"].str.contains(" Battles")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Battles"),"name"] = pokedex[pokedex["name"].str.contains(" Battles")]["name"].apply(lambda s: re.split('\s',s,1)[0])

pokedex.loc[pokedex["name"].str.contains(" Shield"),"suffix"] = pokedex[pokedex["name"].str.contains(" Shield")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Shield"),"name"] = pokedex[pokedex["name"].str.contains(" Shield")]["name"].apply(lambda s: re.split('\s',s,1)[0])

pokedex.loc[pokedex["name"].str.contains(" Sword"),"suffix"] = pokedex[pokedex["name"].str.contains(" Sword")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Sword"),"name"] = pokedex[pokedex["name"].str.contains(" Sword")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# 'Eternamax' suffix: All Pokemon with these suffixes have 1 word common names.
pokedex.loc[pokedex["name"].str.contains(" Eternamax"),"suffix"] = pokedex[pokedex["name"].str.contains(" Eternamax")]["name"].apply(set_suffix,table=suffix_encoding)
pokedex.loc[pokedex["name"].str.contains(" Eternamax"),"name"] = pokedex[pokedex["name"].str.contains(" Eternamax")]["name"].apply(lambda s: re.split('\s',s,1)[0])

# The suffixes of the Pokemon 'Hoopa Hoopa': Indices: 844, 845 ;
# "Hoopa Hoopa" has 2 words in the common name between its variants. Hence, we need to handle its suffixes seperately.
# The suffixes are: "Unbound" and "Confined":
suffix_encoding["Confined"] = suffix_encoding.get("Confined",suffix_encoding.size)
suffix_encoding["Unbound"] = suffix_encoding.get("Unbound",suffix_encoding.size)

pokedex.loc[844,"name"] = pokedex.loc[845,"name"] = "Hoopa Hoopa"
pokedex.loc[844,"suffix"] = suffix_encoding["Confined"]
pokedex.loc[845,"suffix"] = suffix_encoding["Unbound"] 
```


```python
# Encoding remaining Prefixes:

# The 'Partner' prefix:
prefix_encoding["Partner"] = prefix_encoding.get("Partner",prefix_encoding.size)
# The 'Tapu' prefix:
prefix_encoding["Tapu"] = prefix_encoding.get("Tapu",prefix_encoding.size)
# The 'Primal' prefix:
prefix_encoding["Primal"] = prefix_encoding.get("Primal",prefix_encoding.size)
# The 'Black' and 'White' prefixes:
prefix_encoding["Black"] = prefix_encoding.get("Black",prefix_encoding.size)
prefix_encoding["White"] = prefix_encoding.get("White",prefix_encoding.size)
# The 'Ultra' prefix:
prefix_encoding["Ultra"] = prefix_encoding.get("Ultra",prefix_encoding.size)

pokedex = pokedex.apply(set_prefix,axis="columns",table=prefix_encoding)

# EXCEPTIONS: The "Dawn Wings", "Dusk Mane", and "Own Tempo" prefixes: (2 word prefixes)
prefix_encoding["Dawn Wings"] = prefix_encoding.get("Dawn Wings",prefix_encoding.size)
pokedex.loc[935,"name"] = "Necrozma"
pokedex.loc[935,"prefix"] = prefix_encoding["Dawn Wings"]
prefix_encoding["Dusk Mane"] = prefix_encoding.get("Dusk Mane",prefix_encoding.size)
pokedex.loc[934,"name"] = "Necrozma"
pokedex.loc[934,"prefix"] = prefix_encoding["Dusk Mane"]
prefix_encoding["Own Tempo"] = prefix_encoding.get("Own Tempo",prefix_encoding.size)
pokedex.loc[873,"name"] = "Rockruff"
pokedex.loc[873,"prefix"] = prefix_encoding["Own Tempo"]

# All the Pokemon with the above prefixes, have a 1 word common name.

```


```python
# Closing analysis of 'name' of a Pokemon:

# Now, after having removed Prefixes and Suffixes, can we say that the 'name' 
# is the unique identifier of the 'pokedex_number', and vice-versa?

# Does there exist any 'pokedex_number' which corresponds to more than 1 'name'. (YES, ONLY Counter Example: 658)
print("Does any exception exist:",(pokedex.groupby("pokedex_number").apply(lambda df: df["name"].unique().size).sort_values(ascending=False)>1).any())

# Does there exist any 'name' which corresponds to more than 1 'pokedex_number'. (NO!)
print("Does any exception exist:",(pokedex.groupby("name").apply(lambda df: df["pokedex_number"].unique().size).sort_values(ascending=False)>1).any())

# Barring the ONLY exception, 'pokedex_number' 658, we the residual 'name' of a Pokemon 
# holds a unique 1-to-1 relation to the 'pokedex_number' column.
print(pokedex[pokedex["pokedex_number"]==658].iloc[:,[1,2,3,4,5,45,46]])

# Let us handle this exception by giving Pokemon 771 a suffix to represent its "Ash"-ness.
suffix_encoding["Ashy"] = suffix_encoding.get("Ashy",suffix_encoding.size)
pokedex.loc[771,"name"] = "Greninja"
pokedex.loc[771,"suffix"] = suffix_encoding["Ashy"]

# Does there exist any 'pokedex_number' which corresponds to more than 1 'name' now?
print("Does any exception exist now:",(pokedex.groupby("pokedex_number").apply(lambda df: df["name"].unique().size).sort_values(ascending=False)>1).any())

# Hence, we have found the bijective relation between 'pokedex_number' and our modified 'name', which holds for all known Pokemon in existence.
# Let us use this relation to reduce the complexity of Pokemon by Encoding the 'name's by this relation.
name_encoding = pokedex.groupby("name").apply(lambda df: max(df["pokedex_number"].unique()))
name_encoding["Null"] = 0 # To match Pokedex Numbers with encoding indices.
name_encoding.sort_values(inplace=True)

# No further analysis on the 'name' of a Pokemon:
pokedex.drop(columns=["name"],inplace=True,errors="ignore")

# Hence our intuition of "similar" Pokemon names unveiled the underlying relation that 'name' held with 'pokedex_number'.
name_encoding.head(10)
```

    Does any exception exist: True
    Does any exception exist: False
                 name  generation  status  species  type_1  prefix  suffix
    770      Greninja           6       1      198      18       0       0
    771  Ash-Greninja           6       1      198      18       0       0
    Does any exception exist now: False





    name
    Null          0
    Bulbasaur     1
    Ivysaur       2
    Venusaur      3
    Charmander    4
    Charmeleon    5
    Charizard     6
    Squirtle      7
    Wartortle     8
    Blastoise     9
    dtype: int64




```python
# Reordering 'prefix' and 'suffix' columns:
new_order = ['prefix', 'pokedex_number', 'suffix', 'generation', 'status', 'species', 'type_1', 'type_2',
       'height_m', 'weight_kg', 'ability_1', 'ability_2', 'ability_hidden',
       'total_points', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense',
       'speed', 'catch_rate', 'base_friendship', 'base_experience',
       'growth_rate', 'egg_type_1', 'egg_type_2', 'percentage_male',
       'egg_cycles', 'D13', 'D7', 'D18', 'D4', 'D10', 'D12', 'D6', 'D14',
       'D11', 'D8', 'D15', 'D1', 'D16', 'D9', 'D3', 'D2', 'D17', 'D5',]
pokedex = pokedex.loc[:,new_order]
```


```python
pokedex.apply(desc_col,axis="index").T.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Count</th>
      <th>Missing</th>
      <th>Unique Count</th>
      <th>Data Type</th>
      <th>Mean</th>
      <th>Deviation</th>
      <th>Minimum</th>
      <th>Maximum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>prefix</td>
      <td>1045</td>
      <td>0</td>
      <td>18</td>
      <td>int64</td>
      <td>0.324402</td>
      <td>1.532855</td>
      <td>0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pokedex_number</td>
      <td>1045</td>
      <td>0</td>
      <td>898</td>
      <td>int64</td>
      <td>440.769378</td>
      <td>262.517231</td>
      <td>1</td>
      <td>898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>suffix</td>
      <td>1045</td>
      <td>0</td>
      <td>66</td>
      <td>int64</td>
      <td>2.550239</td>
      <td>10.225082</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>generation</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>int64</td>
      <td>4.098565</td>
      <td>2.272788</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>status</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>int64</td>
      <td>1.219139</td>
      <td>0.651271</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>species</td>
      <td>1045</td>
      <td>0</td>
      <td>652</td>
      <td>int64</td>
      <td>285.697608</td>
      <td>198.263458</td>
      <td>0</td>
      <td>651</td>
    </tr>
    <tr>
      <th>6</th>
      <td>type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>18</td>
      <td>int64</td>
      <td>10.407656</td>
      <td>5.524097</td>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>type_2</td>
      <td>1045</td>
      <td>0</td>
      <td>19</td>
      <td>int64</td>
      <td>5.041148</td>
      <td>5.827109</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>height_m</td>
      <td>1045</td>
      <td>0</td>
      <td>61</td>
      <td>float64</td>
      <td>1.374067</td>
      <td>3.353349</td>
      <td>0.1</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>weight_kg</td>
      <td>1044</td>
      <td>1</td>
      <td>474</td>
      <td>float64</td>
      <td>71.216571</td>
      <td>132.259911</td>
      <td>0.1</td>
      <td>999.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ability_1</td>
      <td>1045</td>
      <td>0</td>
      <td>213</td>
      <td>int64</td>
      <td>137.22201</td>
      <td>74.445257</td>
      <td>0</td>
      <td>265</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ability_2</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>70.616268</td>
      <td>87.593765</td>
      <td>0</td>
      <td>262</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ability_hidden</td>
      <td>1045</td>
      <td>0</td>
      <td>155</td>
      <td>int64</td>
      <td>111.911005</td>
      <td>89.341561</td>
      <td>0</td>
      <td>266</td>
    </tr>
    <tr>
      <th>13</th>
      <td>total_points</td>
      <td>1045</td>
      <td>0</td>
      <td>216</td>
      <td>int64</td>
      <td>439.35311</td>
      <td>121.992897</td>
      <td>175</td>
      <td>1125</td>
    </tr>
    <tr>
      <th>14</th>
      <td>hp</td>
      <td>1045</td>
      <td>0</td>
      <td>103</td>
      <td>int64</td>
      <td>70.067943</td>
      <td>26.671411</td>
      <td>1</td>
      <td>255</td>
    </tr>
    <tr>
      <th>15</th>
      <td>attack</td>
      <td>1045</td>
      <td>0</td>
      <td>123</td>
      <td>int64</td>
      <td>80.476555</td>
      <td>32.432728</td>
      <td>5</td>
      <td>190</td>
    </tr>
    <tr>
      <th>16</th>
      <td>defense</td>
      <td>1045</td>
      <td>0</td>
      <td>114</td>
      <td>int64</td>
      <td>74.670813</td>
      <td>31.259462</td>
      <td>5</td>
      <td>250</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sp_attack</td>
      <td>1045</td>
      <td>0</td>
      <td>119</td>
      <td>int64</td>
      <td>73.031579</td>
      <td>32.745857</td>
      <td>10</td>
      <td>194</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sp_defense</td>
      <td>1045</td>
      <td>0</td>
      <td>107</td>
      <td>int64</td>
      <td>72.298565</td>
      <td>28.098943</td>
      <td>20</td>
      <td>250</td>
    </tr>
    <tr>
      <th>19</th>
      <td>speed</td>
      <td>1045</td>
      <td>0</td>
      <td>127</td>
      <td>int64</td>
      <td>68.807656</td>
      <td>30.210094</td>
      <td>5</td>
      <td>200</td>
    </tr>
    <tr>
      <th>20</th>
      <td>catch_rate</td>
      <td>1045</td>
      <td>0</td>
      <td>36</td>
      <td>float64</td>
      <td>91.141627</td>
      <td>76.183957</td>
      <td>0.0</td>
      <td>255.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>base_friendship</td>
      <td>930</td>
      <td>115</td>
      <td>7</td>
      <td>float64</td>
      <td>64.064516</td>
      <td>21.452532</td>
      <td>0.0</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>base_experience</td>
      <td>925</td>
      <td>120</td>
      <td>180</td>
      <td>float64</td>
      <td>153.716757</td>
      <td>79.28397</td>
      <td>36.0</td>
      <td>608.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>growth_rate</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>int64</td>
      <td>4.196172</td>
      <td>1.084895</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>24</th>
      <td>egg_type_1</td>
      <td>1045</td>
      <td>0</td>
      <td>16</td>
      <td>int64</td>
      <td>7.164593</td>
      <td>3.799033</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>25</th>
      <td>egg_type_2</td>
      <td>1045</td>
      <td>0</td>
      <td>12</td>
      <td>int64</td>
      <td>2.873684</td>
      <td>4.926989</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>26</th>
      <td>percentage_male</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>45.789474</td>
      <td>27.631585</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>egg_cycles</td>
      <td>1045</td>
      <td>0</td>
      <td>11</td>
      <td>float64</td>
      <td>31.479392</td>
      <td>30.49825</td>
      <td>5.0</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>D13</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.86866</td>
      <td>0.286863</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>D7</td>
      <td>1045</td>
      <td>0</td>
      <td>8</td>
      <td>float64</td>
      <td>1.126316</td>
      <td>0.714569</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>D18</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.050718</td>
      <td>0.609383</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>D4</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.035646</td>
      <td>0.643535</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>D10</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.001196</td>
      <td>0.74435</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>D12</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.675598</td>
      <td>7.685308</td>
      <td>0.0</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>D6</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.07512</td>
      <td>0.753649</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>D14</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.95311</td>
      <td>0.541238</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>D11</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.082297</td>
      <td>0.782683</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>D8</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.1689</td>
      <td>0.592145</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>D15</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>0.977273</td>
      <td>0.501934</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>D1</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.998086</td>
      <td>0.610411</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>D16</td>
      <td>1045</td>
      <td>0</td>
      <td>5</td>
      <td>float64</td>
      <td>1.238278</td>
      <td>0.69656</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D9</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.01866</td>
      <td>0.568056</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>D3</td>
      <td>1045</td>
      <td>0</td>
      <td>4</td>
      <td>float64</td>
      <td>0.977033</td>
      <td>0.375812</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>D2</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>1.071053</td>
      <td>0.465178</td>
      <td>0.25</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>D17</td>
      <td>1045</td>
      <td>0</td>
      <td>7</td>
      <td>float64</td>
      <td>0.981579</td>
      <td>0.501753</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>D5</td>
      <td>1045</td>
      <td>0</td>
      <td>6</td>
      <td>float64</td>
      <td>1.091148</td>
      <td>0.536285</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
