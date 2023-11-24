### Importing necessary modules:


```python
# These Python modules are necessary for processing the data and also for visualisations.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import NN
import losses
```

### Study of Wake effect as described by the N. O. Jensen model:

#### N. O. Jensen Single Wake effect model:

##### Defining a Wind Turbine:


```python
class WindTurbine(object):
    def __init__(self, rotor_D, hub_h, Ct, alpha, x, y, V, hill_x = 0, hill_y = 0, H = 0, L = 1) -> None:
        self.rotor_D = rotor_D # Rotor diameter (m)
        self.hub_h = hub_h # Hub-height (m)

        # Position of Turbine in plane: (x +ve downstream)
        self.x = x
        self.y = y

        # The hill coordinates of the turbine:
        self.hl_x = hill_x
        self.hl_y = hill_y

        # The bell-coefficients for the hill:
        self.hl_H = H
        self.hl_L = L
        self.hill_coeff()

        # The ground-wind speed:
        self.wind = V

        # Initial undisturbed upstream wind experienced by Turbine blades
        self.V = self.wind*(1 + 4*self.hl_a*self.hl_s)

        self.Ct = Ct # Coefficient of thrust
        self.alpha = alpha # Linear wake expansion factor
        

    def hill_coeff(self):
        h = self.hl_H / (1 + ((self.hl_y-self.y)/self.hl_L)**2)
        if(h==0):
            self.hl_s = 0
            self.hl_a = 1
            return
        
        l = self.hl_L * self.hl_H / (h if h!=0 else 1)
        if(l==0):
            self.hl_s = 0
            self.hl_a = 1
            return


        self.hl_s = h / (2*l)
        self.hl_a = (1 - np.abs(self.hl_x - self.x)/(1.5*l))*(np.exp(-2.5*self.hub_h/l) if (self.hl_s<0.3) else np.exp(-1.5*self.hub_h/h))
        return

```

##### The N. O. Jensen Wake Deficit Model: (with Terrain Effect)


```python
def jensen_model_deficit(WT: WindTurbine, x_down, ):
    """
    N. O. Jensen model for wake velocity deficit

    Parameters: (vectorized)
    V0 : float
        Free stream wind speed (m/s)
    rotor_D : float
        Rotor diameter of the turbine (m)
    alpha: float
        Wake expansion factor
    x_down : float
        Distance downstream from the turbine (m)
    Ct : float
        Rotor thrust coefficient (m)

    Returns: (vectorized)
    float
        Wake velocity deficit at different distances behind the turbine (m/s)
    """
    V0, Ct, rotor_D, alpha, a, s, = WT.wind, WT.Ct, WT.rotor_D, WT.alpha, WT.hl_a, WT.hl_s

    rotor_r = rotor_D/2
    wake_r = rotor_r + alpha*x_down

    wake_deficit = np.where(wake_r>=rotor_r,V0*(1-np.sqrt(1-Ct))*np.power(rotor_r/wake_r,2),0) # If 'x_down' is negative 'Wake deficit' is not defined (0)

    # Lissaman model with terrain effect:
    wake_deficit = wake_deficit / (1 + 4*a*s)
    
    return wake_deficit
```

##### Testing the Single Wake model:


```python
# Parameters:

V = float(input("Free stream wind speed (m/s):")) # Free stream wind speed (m/s)
rotor_D = float(input("Enter rotor diameter(m):"))   # Rotor diameter of the turbine (m)

hub_h = int(input("Hub-height:")) # Hub height of the turbine
z0 = float(input("Surface roughness factor:"))   # Surface roughness coefficient
alpha = 1/(2*np.log(hub_h/z0))
alpha = 0.04 # Common value

Ct = float(input("Enter the thrust coefficient:")) # Thrust coefficient of turbine

# Defining the Wind Turbine
WT = WindTurbine(rotor_D, hub_h, Ct, alpha, 0, 0, V, 1,2,50,30)

WT1 = WindTurbine(rotor_D, hub_h, Ct, alpha, 0, 0, V)

# Distance range
x_range = np.linspace(0, 100 * WT.rotor_D, 100)

# Calculate the wake velocity deficit
vel_deficit = jensen_model_deficit(WT, x_range,)
vel_ = jensen_model_deficit(WT1,x_range)
wake_velocity = WT.V - vel_deficit

# Plot the wake velocity and deficit
# plt.plot(x_range, wake_velocity, label='Wind Velocity')
plt.plot(x_range, vel_deficit, label='Wake Velocity Deficit (with Terrain effect)')
plt.plot(x_range,vel_,label='Wake Velocity Deficit (without Terrain effect)')
plt.xlabel('Distance Downstream from Turbine (m)')
plt.ylabel('Velocity (m/s)')
plt.title('N. O. Jensen Model - Wake Effect of a Single Turbine')
plt.legend()
plt.grid()
plt.show()

```


    
![png](output_9_0.png)
    


#### Multiple Wake Effect on a Wind Farm using Superposition Principle:

##### Implementing Multiple Wake Superposition and Area Overlap modelling methods:


```python
def calc_shadow_coeff(WT1:WindTurbine,WT2:WindTurbine):
    """
    Calculate the area of shadow cast on the area swept by turbine WT2, 
    by the Wake circle of the turbine WT1.

    Returns the fractional portion of WT2 area covered.
    """
    
    # As shadow is only cast by upstream turbine on downstream turbine
    if(WT1.x >= WT2.x):
        return 0
    
    dx_12 = WT2.x - WT1.x

    R = WT1.rotor_D/2 + WT1.alpha * dx_12
    r = WT2.rotor_D/2

    R, r = max(R,r), min(R,r)

    # Considering terrain effect and hub heights
    H1 = WT1.hl_H / (1 + ((WT1.x-WT1.hl_x)/WT1.hl_L)**2)
    H2 = WT2.hl_H / (1 + ((WT2.x-WT2.hl_x)/WT2.hl_L)**2)
    dy_12 = np.sqrt( (WT2.y - WT1.y)**2 + ((WT1.hub_h+H1) - (WT2.hub_h+H2))**2 )


    if(dy_12<=np.abs(R-r)): # Full overlap:
        return 1
    
    if(dy_12>=R+r): # No overlap:
        return 0
    
    # Partial overlap:
    h_center = (R + r + dy_12) / 2
    A_tri = np.sqrt((h_center)*(h_center-R)*(h_center-r)*(h_center-dy_12))

    l = 2 * A_tri / dy_12
    cos_theta = (R**2+r**2-dy_12**2)/(2*R*r)
    if (cos_theta>r/R):
        A_shadow = (R**2)*np.arctan(l/R) + (r**2)*(np.pi - np.arctan(l/r)) - 2*A_tri
    else:
        A_shadow = (R**2)*np.arctan(l/R) + (r**2)*(np.arctan(l/r)) - 2*A_tri
    
    # Return the fraction of total area shadowed:
    return A_shadow / (np.pi*((WT2.rotor_D/2)**2))


def calc_WT_vel_superposition(WT:WindTurbine,turbines):
    """
    Calculates and updates the net velocity experienced by turbine WT, 
    from upstream, owing to the cumulative Wake effect of all 
    upstream Wind Turbines, calculated using momentum-conserved superposition.
    """
    total_deficit = 0
    for wt in turbines:
        if(wt.x>WT.x):
            break
        
        beta = calc_shadow_coeff(wt,WT)

        # Very inefficient implementation
        total_deficit += beta*(WT.V**2 - (WT.V - jensen_model_deficit(wt, WT.x-wt.x)*beta)**2)
        
        # print(f"({WT.x,WT.y}) Shadow_coeff: {beta}")
    
    # Updating the effective velocity of oncoming upstream wind, after superposition of Wake effects.
    WT.V = np.sqrt(WT.V**2 - total_deficit)
    
    return

def calc_point_vel_superposition(V,x,y,turbines):
    """
    Calculate the net wind speed at a point in the plane, 
    due to the cumulative effects of all upstream Wakes.

    Parameters:

    x : float
        The x co-ordinate of the point in plane
    y : float
        The y co-ordinate of the point in plane
    V : float
        The undisturbed wind speed at point.
    turbines : array of WindTurbine elements
        The description of all WT's in the farm
    """
    vel_deficit = 0
    for wt in turbines:
        if(wt.x>x):
            break
        if(np.abs(wt.y - y) <= (wt.rotor_D/2 + wt.alpha*(x - wt.x))):
            # Very inefficient implementation
            vel_deficit += (V**2 - (V - jensen_model_deficit(wt, x-wt.x))**2)
     
    res = np.sqrt(V**2 - vel_deficit)

    return res

```

##### Defining the Power Curve of the Wind Turbine:


```python
def power_out(vel, rated_p=1.8, cut_in_v = 4, cut_out_v = 25, rated_v = 13):
    """
    Returns the power output of a WT (mega watts) given upstream wind speed 'vel' (m/s)
    Default values are chosen of the Vestas V90 1.8 MW Wind Turbine
    """
    return np.where(vel>=cut_in_v ,np.where(vel<rated_v ,rated_p*(vel**3-cut_in_v**3)/(rated_v**3-cut_in_v**3),np.where(vel<=cut_out_v,rated_p,0)),0)
```

##### Defining the Wind Farm layout and Wind Turbine specifications:


```python
# Description of the Wind Farm layout and Turbine specification:

# WT specifications
rotor_D, hub_h, Ct, alpha = 90, 100, 0.3, 0.08

# Wind Farm layout:
V = 8.5
theta = 0
seperation  = 7 * rotor_D
rows = 10
columns = 6

# Initializing WT's according to layout and specifications.
turbines = np.empty(shape=(rows,columns),dtype=np.dtype("object"))
for i in range(rows):
    for j in range(columns):

        # Assigning the position of WT on plane (Grid configuration)
        x_co = seperation*j
        y_co = seperation*i
        
        # # Another configuration (Alternate Shifted Grid)
        # x_co = seperation*j 
        # y_co = seperation*i + seperation*(j%2)/2
        
        turbines[i,j] = WindTurbine(rotor_D,hub_h,Ct,alpha,x_co,y_co,V)

turbines = turbines.T.flatten() # Change the layout of the array from 2D to 1D.
```

##### Defining function for Rotation of Plane and Calculation of Velocity Field:


```python
def rotate_turbine_layout(turbines, theta):
    """
    Rotates the plane of Wind Turbines to align wind direction along +ve x-axis.

    Input:
        'theta': Angle by which plane is rotated w.r.t. +ve x-axis.
        'turbines' : A collection of all Wind Turbines in the farm, each at a particular point in 2D-plane.    
    """
    theta = theta*np.pi/180
    for wt in turbines:
        wt.x, wt.y = (wt.x)*np.cos(theta) - (wt.y)*np.sin(theta) , (wt.x)*np.sin(theta) + (wt.y)*np.cos(theta)

        # Adjust terrain effect:
        wt.hl_x, wt.hl_y = (wt.hl_x)*np.cos(theta) - (wt.hl_y)*np.sin(theta) , (wt.hl_x)*np.sin(theta) + (wt.hl_y)*np.cos(theta)
        wt.hill_coeff() # Set the new coefficients
        wt.V = wt.wind * (1 + 4*wt.hl_a*wt.hl_s) # Update the upstream velocity
    
    return np.asarray(sorted(turbines,key=lambda WT: WT.x))

def WF_vel_distribution(turbines):
    # Calculating the upper and lower limits of the plane:
    y_upper = y_lower = 0
    for wt in turbines:
        if(wt.y<y_lower):
            y_lower = wt.y
        if(wt.y>y_upper):
            y_upper = wt.y
    x_lower = turbines[0].x
    x_upper = turbines[rows*columns - 1].x

    # Defining the plane on which wind velocity field will be calculated.
    x_range = np.linspace(x_lower - 10, x_upper + seperation, 200)
    y_range = np.linspace(y_lower - 3*rotor_D, y_upper + 3*rotor_D, 200)

    # Populating the planar space with wind calculated velocities
    def vel_field(turbines,x_range,y_range):
        dist = np.empty(shape=(x_range.size,y_range.size),dtype=np.dtype("float64"))
        for x_i in range(x_range.size):
            for y_i in range(y_range.size):
                dist[x_i,y_i] = calc_point_vel_superposition(V,x_range[x_i],y_range[y_i], turbines)

        return dist.T
    
    return vel_field(turbines,x_range, y_range)

```

##### Computation of Wake effect and Farm Power Output:


```python
# Test the Wind Farm for different Wind Directions:
velocity_fields = []
wind_angle = []
generated_power = []

step_number = 18
step_angle = 180//step_number

for i in range(step_number):
    # Set the idle wind-velocity faced by each turbine:
    for wt in turbines:
        wt.V = V

    # Calculating Wake effect over the entire Wind Farm:
    for wt in turbines:
        calc_WT_vel_superposition(wt,turbines)

    total_output = 0
    for wt in turbines:
        total_output += power_out(wt.V)

    velocity_fields.append(WF_vel_distribution(turbines))
    wind_angle.append(theta)
    generated_power.append(total_output)

    # Rotating planar frame, to adjust wind direction:
    turbines = rotate_turbine_layout(turbines,-(step_angle)) # Rotation of wind direction along +ve x-axis, hence opposite(-) rotation aligns layout.

    theta = (theta + step_angle)%360

    # for wt in angled_turbines:
    #     print(f"Position:({wt.x},{wt.y}) Velocity: {wt.V} m/s")


```

##### Visualising the Wind Velocity Fields for different Wind Conditions:


```python
def plot_contous(velocity_fields,rotations,power_outs):
    """
    Plot contour plots for each 2D matrix in a given list of velocity_fields.
    The whole figure will have a single colorbar for all plots with uniform color coding based on the minimum and maximum values across all velocity_fields.

    Parameters:
    velocity_fields (list of 2D arrays): A list of 2D velocity_fields (numpy arrays) to be plotted.

    rotations : A list of angles by which the plane is rotated.

    Returns:
    None
    """

    # Calculate the minimum and maximum values across all velocity_fields
    min_val = min(np.min(matrix) for matrix in velocity_fields)
    max_val = max(np.max(matrix) for matrix in velocity_fields)

    num_velocity_fields = len(velocity_fields)
    num_rows = (num_velocity_fields + 2) // 3  # Number of rows in the figure
    num_cols = min(3 , num_velocity_fields)  # Number of columns in the figure

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15*num_cols, 5*num_rows))

    contour_plots = []  # List to store contour plots for all velocity_fields

    for i, matrix in enumerate(velocity_fields):
        if num_velocity_fields > 1:
            ax = axes[i // num_cols, i % num_cols]
        else:
            ax = axes  # For a single matrix, use the same axis

        # Create a contour plot for the current matrix with uniform color coding
        contour = ax.contourf(matrix, levels=np.linspace(min_val, max_val, 200))

        # Set plot title and labels
        ax.set_title(f"Wind Angle: {rotations[i]}°   Farm Output: {'{:.3f}'.format(power_outs[i])} MW ",fontsize=20)
        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")

        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
        contour_plots.append(contour)

    # Get the maximum number of contour levels among all plots for the colorbar
    # num_levels = max(len(c.levels) for c in contour_plots)

    # Create a single horizontal colorbar for all contour plots with the same number of levels
    cbar = fig.colorbar(contour_plots[0], ax=axes, ticks=np.linspace(min_val, max_val, 10), orientation='vertical',aspect=30)

    # Set the fontsize of the color bar labels
    cbar.ax.tick_params(labelsize=20)

    # Add a title for the entire figure
    # plt.suptitle("Directed Wind Velocity Distributions", fontsize=16)
    
    plt.show()

# Display the results:
plot_contous(velocity_fields,wind_angle,generated_power)
```


    
![png](output_22_0.png)
    


### Study of the Power Curve of a Wind Turbine Generation Characteristics

#### Power output calculations and characteristic curves:

##### Importing and organising the Wind and Power Output data:


```python
path = ".\wind_data.txt"

windpower_data = pd.read_csv(path,sep='\t')
windpower_data.head()
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
      <th>Date(YYYY-MM-DD hh:mm:ss)</th>
      <th>W</th>
      <th>P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2006 00:00</td>
      <td>5.12</td>
      <td>1.788</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-01-2006 00:10</td>
      <td>5.12</td>
      <td>1.788</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-01-2006 00:20</td>
      <td>5.14</td>
      <td>1.812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-01-2006 00:30</td>
      <td>5.17</td>
      <td>1.848</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-01-2006 00:40</td>
      <td>5.24</td>
      <td>1.929</td>
    </tr>
  </tbody>
</table>
</div>




```python
windpower_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 52560 entries, 0 to 52559
    Data columns (total 3 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   Date(YYYY-MM-DD hh:mm:ss)  52560 non-null  object 
     1   W                          52560 non-null  float64
     2   P                          52560 non-null  float64
    dtypes: float64(2), object(1)
    memory usage: 1.2+ MB
    


```python
date_format = "%d-%m-%Y %H:%M"
windpower_data["Date(YYYY-MM-DD hh:mm:ss)"] = pd.to_datetime(windpower_data["Date(YYYY-MM-DD hh:mm:ss)"],format=date_format)
# windpower_data["P"] = windpower_data["P"] / 100 # Converting to MW
```


```python
windpower_data.set_index("Date(YYYY-MM-DD hh:mm:ss)",inplace=True)
windpower_data.rename_axis(index=[None, ],inplace=True)
windpower_data.rename(columns={"W":"Wind Speed (m/s)", "P":"Actual Power Output", },inplace=True)
```


```python
windpower_data.head()
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
      <th>Wind Speed (m/s)</th>
      <th>Actual Power Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-01 00:00:00</th>
      <td>5.12</td>
      <td>1.788</td>
    </tr>
    <tr>
      <th>2006-01-01 00:10:00</th>
      <td>5.12</td>
      <td>1.788</td>
    </tr>
    <tr>
      <th>2006-01-01 00:20:00</th>
      <td>5.14</td>
      <td>1.812</td>
    </tr>
    <tr>
      <th>2006-01-01 00:30:00</th>
      <td>5.17</td>
      <td>1.848</td>
    </tr>
    <tr>
      <th>2006-01-01 00:40:00</th>
      <td>5.24</td>
      <td>1.929</td>
    </tr>
  </tbody>
</table>
</div>



##### Using the previously defined Power Curve to Calculate Output:


```python
# Calculating the theoretical power output at different wind speeds:
windpower_data["Theoretical Power Output (MW)"] = power_out(windpower_data["Wind Speed (m/s)"])
windpower_data.head()
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
      <th>Wind Speed (m/s)</th>
      <th>Actual Power Output</th>
      <th>Theoretical Power Output (MW)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2006-01-01 00:00:00</th>
      <td>5.12</td>
      <td>1.788</td>
      <td>0.059255</td>
    </tr>
    <tr>
      <th>2006-01-01 00:10:00</th>
      <td>5.12</td>
      <td>1.788</td>
      <td>0.059255</td>
    </tr>
    <tr>
      <th>2006-01-01 00:20:00</th>
      <td>5.14</td>
      <td>1.812</td>
      <td>0.060588</td>
    </tr>
    <tr>
      <th>2006-01-01 00:30:00</th>
      <td>5.17</td>
      <td>1.848</td>
      <td>0.062606</td>
    </tr>
    <tr>
      <th>2006-01-01 00:40:00</th>
      <td>5.24</td>
      <td>1.929</td>
      <td>0.067407</td>
    </tr>
  </tbody>
</table>
</div>



##### Visualising the Offshore Wind Speed data:


```python
# The wind speed data over a period of time visualised:
plt.figure(figsize=(12, 6))

start_date = '2006-11-06 00:00'
end_date = '2006-11-07 00:00'
data_subinterval = windpower_data.loc[start_date:end_date]

# plt.plot(data_subinterval.index, data_subinterval['P'], marker='.', linestyle='-')
plt.plot(data_subinterval.index, data_subinterval['Wind Speed (m/s)'])

plt.xlabel('Date')
plt.ylabel('Wind Speed (m/s)')
plt.title('Time Series Data for Sub-Interval')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](output_34_0.png)
    


##### Visualising the Power Curves from given Wind Speed data:


```python
# The Power Curve (Output Power v.s. Wind Speed characteristics)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(15,6),layout="constrained")

sns.scatterplot(windpower_data,x="Wind Speed (m/s)",y="Actual Power Output",hue="Wind Speed (m/s)",palette=sns.color_palette("viridis",as_cmap=True),ax=axs[0])

axs[0].grid()
axs[0].set_ylabel('Actual Output Power')
axs[0].set_xlabel('Wind Speed (m/s)')
# axs[0].set_facecolor("grey")
axs[0].set_title('Power Curve of an Offshore Wind Turbine')

sns.scatterplot(windpower_data,x="Wind Speed (m/s)",y="Theoretical Power Output (MW)",hue="Wind Speed (m/s)",palette=sns.color_palette("viridis",as_cmap=True),ax=axs[1])

axs[1].grid()
axs[1].set_ylabel('Theoretical Output Power (MW)')
axs[1].set_xlabel('Wind Speed (m/s)')
# axs[1].set_facecolor("grey")
axs[1].set_title('Power Curve of the Vestas V90 1.8MW Wind Turbine')

plt.show()
```


    
![png](output_36_0.png)
    


##### Visualising the Variation of Farm Output with Wind Direction:


```python
deg_res = 0.8
output_curve = []

# theta = 0
for i in range(int(360/deg_res)):
    # Set the idle wind-velocity faced by each turbine:
    for wt in turbines:
        wt.V = V

    # Calculating Wake effect over the entire Wind Farm:
    for wt in turbines:
        calc_WT_vel_superposition(wt,turbines)    

    total_output = 0
    for wt in turbines:
        total_output += power_out(wt.V)

    output_curve.append([theta, total_output])

    # Rotating planar frame, to adjust wind direction:
    turbines = rotate_turbine_layout(turbines,-(deg_res)) # Rotation of wind direction along +ve x-axis, hence opposite(-) rotation aligns layout.

    theta = (theta + deg_res)%360

output_curve.sort(key=lambda x: x[0])
output_curve = np.asarray(output_curve).T

avg_farm_out = np.sum(output_curve[1]) / output_curve.shape[1]
```


```python
def plot_points(x_coords, y_coords, line=True,marks=False,avg_farm_out = 0):
    """
    Plot a list of points on a 2D coordinate plane.

    Parameters:
    'x_coords' and 'y_coords' : Arrays which together describe each point in 2D plane.

    """
    plt.figure(figsize=(15,8))

    # Create the plot with scatter points
    if(line and marks):
        plt.plot(x_coords, y_coords, color='red', marker='.', linestyle='-', linewidth=1)
    elif(line):
        plt.plot(x_coords, y_coords, color='red', linestyle='-', linewidth=1)
    else:
        plt.plot(x_coords, y_coords, color='red', marker='.', linestyle='')

    plt.xlabel("Wind Angle (°)")
    plt.ylabel("Farm Output Power (MW)")
    plt.title(f"Variation in Farm Output with Wind Direction (Average Farm Output Power: {avg_farm_out} MW)")
    plt.grid(True)

    # Show the plot
    plt.show()

plot_points(output_curve[0],output_curve[1],line=True,marks=False,avg_farm_out=avg_farm_out)
```


    
![png](output_39_0.png)
    


##### Power Curve of the Vestas V90 2MW Wind Turbine:


```python
path = "./Vestas 2MW WT.txt"
vestas_2mw_data = pd.read_csv(path,sep=' ',index_col=0)

# Removing the 'index' title:
vestas_2mw_data.rename_axis(index=None,inplace=True)

# Sorting the columns:
vestas_2mw_data.rename(lambda x: float(x),axis="columns",inplace=True)
vestas_2mw_data.sort_index(axis="columns",inplace=True)

vestas_2mw_data.head()
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
      <th>0.950</th>
      <th>0.975</th>
      <th>1.000</th>
      <th>1.025</th>
      <th>1.050</th>
      <th>1.075</th>
      <th>1.100</th>
      <th>1.125</th>
      <th>1.150</th>
      <th>1.175</th>
      <th>1.200</th>
      <th>1.225</th>
      <th>1.250</th>
      <th>1.275</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4.0</th>
      <td>63</td>
      <td>66</td>
      <td>68</td>
      <td>70</td>
      <td>73</td>
      <td>75</td>
      <td>78</td>
      <td>80</td>
      <td>82</td>
      <td>85</td>
      <td>87</td>
      <td>89</td>
      <td>92</td>
      <td>94</td>
    </tr>
    <tr>
      <th>4.5</th>
      <td>105</td>
      <td>108</td>
      <td>112</td>
      <td>115</td>
      <td>119</td>
      <td>122</td>
      <td>125</td>
      <td>129</td>
      <td>132</td>
      <td>136</td>
      <td>139</td>
      <td>142</td>
      <td>146</td>
      <td>149</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>154</td>
      <td>158</td>
      <td>163</td>
      <td>167</td>
      <td>172</td>
      <td>177</td>
      <td>181</td>
      <td>186</td>
      <td>191</td>
      <td>195</td>
      <td>200</td>
      <td>204</td>
      <td>209</td>
      <td>213</td>
    </tr>
    <tr>
      <th>5.5</th>
      <td>211</td>
      <td>217</td>
      <td>224</td>
      <td>230</td>
      <td>236</td>
      <td>242</td>
      <td>248</td>
      <td>254</td>
      <td>260</td>
      <td>266</td>
      <td>272</td>
      <td>279</td>
      <td>285</td>
      <td>291</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>280</td>
      <td>288</td>
      <td>296</td>
      <td>304</td>
      <td>312</td>
      <td>320</td>
      <td>328</td>
      <td>336</td>
      <td>344</td>
      <td>352</td>
      <td>360</td>
      <td>368</td>
      <td>376</td>
      <td>383</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting the column and index labels:
dens = vestas_2mw_data.columns.values
spds = vestas_2mw_data.index.values

# Processing the data:
modified_2mw_data = pd.DataFrame(np.asarray([np.tile(dens,vestas_2mw_data.shape[0]), np.tile(spds,(vestas_2mw_data.shape[1],1)).T.flatten(), vestas_2mw_data.values.flatten()]),index=["Air Density (S.I)", "Wind Speed (S.I)", "Output Power (kW)"])
modified_2mw_data.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>256</th>
      <th>257</th>
      <th>258</th>
      <th>259</th>
      <th>260</th>
      <th>261</th>
      <th>262</th>
      <th>263</th>
      <th>264</th>
      <th>265</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Air Density (S.I)</th>
      <td>0.95</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>1.025</td>
      <td>1.05</td>
      <td>1.075</td>
      <td>1.1</td>
      <td>1.125</td>
      <td>1.15</td>
      <td>1.175</td>
      <td>...</td>
      <td>1.05</td>
      <td>1.075</td>
      <td>1.1</td>
      <td>1.125</td>
      <td>1.15</td>
      <td>1.175</td>
      <td>1.2</td>
      <td>1.225</td>
      <td>1.25</td>
      <td>1.275</td>
    </tr>
    <tr>
      <th>Wind Speed (S.I)</th>
      <td>4.00</td>
      <td>4.000</td>
      <td>4.0</td>
      <td>4.000</td>
      <td>4.00</td>
      <td>4.000</td>
      <td>4.0</td>
      <td>4.000</td>
      <td>4.00</td>
      <td>4.000</td>
      <td>...</td>
      <td>13.00</td>
      <td>13.000</td>
      <td>13.0</td>
      <td>13.000</td>
      <td>13.00</td>
      <td>13.000</td>
      <td>13.0</td>
      <td>13.000</td>
      <td>13.00</td>
      <td>13.000</td>
    </tr>
    <tr>
      <th>Output Power (kW)</th>
      <td>63.00</td>
      <td>66.000</td>
      <td>68.0</td>
      <td>70.000</td>
      <td>73.00</td>
      <td>75.000</td>
      <td>78.0</td>
      <td>80.000</td>
      <td>82.00</td>
      <td>85.000</td>
      <td>...</td>
      <td>1978.00</td>
      <td>1982.000</td>
      <td>1986.0</td>
      <td>1990.000</td>
      <td>1994.00</td>
      <td>1996.000</td>
      <td>1997.0</td>
      <td>1998.000</td>
      <td>1999.00</td>
      <td>1999.000</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 266 columns</p>
</div>



#### Modelling the Power Curve of the Vestas V90 2MW Turbine:

##### The Polynomial Feature Transform:


```python
def polynomial_transform(features:np.ndarray, degree, p_table, pos = 0, res=1):
    """
    Takes a set of 'features', and extracts the polynomial transform of features upto given 'degree'.

    Input:
        features: An ndarray of feature values, which are to be transformed.

        degree: The degree of the polynomial with given 'features' as factors.

    """
    if(pos==features.shape[0] or degree==0):
        # print("Result:",res)
        p_table.append(np.copy(res))
        return p_table

    p = np.ones((features.shape[1], ))*res
    for i in range(degree+1):
        polynomial_transform(features,degree-i,p_table,pos+1,p)
        p *= features[pos]

    return p_table
    
```

##### A 2-variable Polynomial Model considering variable 'Wind Speed' and 'Air density':  


```python
# Feature Engineering: (Polynomial Transform degree-6 with 2-features)
X_train = []
X_train = np.asarray(polynomial_transform(modified_2mw_data.loc[['Air Density (S.I)','Wind Speed (S.I)'],:].values,6,X_train))

# Extracting Data Point Labels:
Y_train = modified_2mw_data.loc["Output Power (kW)",:].values.reshape(1, modified_2mw_data.shape[1])
```


```python
# Defining the model: (degree-6 features-2)
deg6_2mw_polynomial_model = NN(1,[X_train.shape[0], 1, ],["None", ], losses.mse_loss,losses.mse_loss_deriv,AdaM_opt=True)
```


```python
# Hyperparameter tuning:
deg6_2mw_polynomial_model.alpha = 0.0000006

deg6_2mw_polynomial_model.reglr_type = "L1"
deg6_2mw_polynomial_model.lmbd = 0

# To counter-act a bug in code:
deg6_2mw_polynomial_model.frwrd_trials = 10000
deg6_2mw_polynomial_model.back_trials = 10000

# Training:
total_trials = 100001
for trial in range(total_trials):
    train_loss = (NN.training_cycle(deg6_2mw_polynomial_model, X=X_train, Y=Y_train))
    if((trial)%100==0):
        print("Trial number:",trial)
        print("The total training accuracy: ",train_loss)
        print("---------------------------------------------------------------------------------------------------")
```

    Trial number: 0
    The total training accuracy:  [147.77149887]
    ---------------------------------------------------------------------------------------------------
    Trial number: 100
    The total training accuracy:  [147.77120303]
    ---------------------------------------------------------------------------------------------------
    Trial number: 200
    The total training accuracy:  [147.77091003]
    ---------------------------------------------------------------------------------------------------
    Trial number: 300
    The total training accuracy:  [147.77060459]
    ---------------------------------------------------------------------------------------------------
    Trial number: 400
    The total training accuracy:  [147.80154503]
    ---------------------------------------------------------------------------------------------------
    Trial number: 500
    The total training accuracy:  [147.77001201]
    ---------------------------------------------------------------------------------------------------
    Trial number: 600
    The total training accuracy:  [147.76972479]
    ---------------------------------------------------------------------------------------------------
    Trial number: 700
    The total training accuracy:  [147.76942805]
    ---------------------------------------------------------------------------------------------------
    Trial number: 800
    The total training accuracy:  [147.76911869]
    ---------------------------------------------------------------------------------------------------
    Trial number: 900
    The total training accuracy:  [147.77049493]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1000
    The total training accuracy:  [147.76853418]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1100
    The total training accuracy:  [147.76824576]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1200
    The total training accuracy:  [147.76794521]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1300
    The total training accuracy:  [147.76763189]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1400
    The total training accuracy:  [147.76737852]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1500
    The total training accuracy:  [147.76705815]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1600
    The total training accuracy:  [147.76676643]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1700
    The total training accuracy:  [147.76646233]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1800
    The total training accuracy:  [147.76624415]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1900
    The total training accuracy:  [147.76586991]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2000
    The total training accuracy:  [147.76558112]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2100
    The total training accuracy:  [147.76528543]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2200
    The total training accuracy:  [147.7649772]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2300
    The total training accuracy:  [147.77104571]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2400
    The total training accuracy:  [147.76439064]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2500
    The total training accuracy:  [147.76410302]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2600
    The total training accuracy:  [147.76380321]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2700
    The total training accuracy:  [147.76349066]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2800
    The total training accuracy:  [147.76336623]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2900
    The total training accuracy:  [147.76291491]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3000
    The total training accuracy:  [147.76262235]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3100
    The total training accuracy:  [147.76231739]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3200
    The total training accuracy:  [147.80504821]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3300
    The total training accuracy:  [147.76172559]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3400
    The total training accuracy:  [147.76143842]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3500
    The total training accuracy:  [147.76114168]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3600
    The total training accuracy:  [147.76083233]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3700
    The total training accuracy:  [147.76253069]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3800
    The total training accuracy:  [147.76024931]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3900
    The total training accuracy:  [147.75995975]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4000
    The total training accuracy:  [147.75965796]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4100
    The total training accuracy:  [147.75934335]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4200
    The total training accuracy:  [147.75909543]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4300
    The total training accuracy:  [147.75877598]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4400
    The total training accuracy:  [147.75848409]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4500
    The total training accuracy:  [147.75817981]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4600
    The total training accuracy:  [147.75970824]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4700
    The total training accuracy:  [147.75758931]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4800
    The total training accuracy:  [147.75730171]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4900
    The total training accuracy:  [147.75700654]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5000
    The total training accuracy:  [147.75669883]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5100
    The total training accuracy:  [147.76874746]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5200
    The total training accuracy:  [147.75611409]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5300
    The total training accuracy:  [147.75583029]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5400
    The total training accuracy:  [147.7555345]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5500
    The total training accuracy:  [147.75522615]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5600
    The total training accuracy:  [147.75548006]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5700
    The total training accuracy:  [147.75464376]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5800
    The total training accuracy:  [147.75435937]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5900
    The total training accuracy:  [147.75406307]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6000
    The total training accuracy:  [147.75375419]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6100
    The total training accuracy:  [147.75596829]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6200
    The total training accuracy:  [147.753173]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6300
    The total training accuracy:  [147.75288652]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6400
    The total training accuracy:  [147.75258798]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6500
    The total training accuracy:  [147.75227676]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6600
    The total training accuracy:  [147.7520314]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6700
    The total training accuracy:  [147.75170329]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6800
    The total training accuracy:  [147.75141662]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6900
    The total training accuracy:  [147.75111783]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7000
    The total training accuracy:  [147.75080636]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7100
    The total training accuracy:  [147.75053202]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7200
    The total training accuracy:  [147.7502337]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7300
    The total training accuracy:  [147.74994626]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7400
    The total training accuracy:  [147.74964664]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7500
    The total training accuracy:  [147.74933431]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7600
    The total training accuracy:  [147.74904754]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7700
    The total training accuracy:  [147.74876367]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7800
    The total training accuracy:  [147.74847426]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7900
    The total training accuracy:  [147.7481726]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8000
    The total training accuracy:  [147.74785817]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8100
    The total training accuracy:  [147.74759449]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8200
    The total training accuracy:  [147.74729234]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8300
    The total training accuracy:  [147.74699939]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8400
    The total training accuracy:  [147.74669403]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8500
    The total training accuracy:  [147.75862023]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8600
    The total training accuracy:  [147.74610495]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8700
    The total training accuracy:  [147.74581893]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8800
    The total training accuracy:  [147.74552121]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8900
    The total training accuracy:  [147.74521086]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9000
    The total training accuracy:  [147.74516477]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9100
    The total training accuracy:  [147.74463412]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9200
    The total training accuracy:  [147.74434351]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9300
    The total training accuracy:  [147.74404061]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9400
    The total training accuracy:  [147.74419574]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9500
    The total training accuracy:  [147.74345063]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9600
    The total training accuracy:  [147.74316174]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9700
    The total training accuracy:  [147.74286588]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9800
    The total training accuracy:  [147.74255749]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9900
    The total training accuracy:  [147.7425107]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10000
    The total training accuracy:  [147.74197629]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10100
    The total training accuracy:  [147.74168693]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10200
    The total training accuracy:  [147.74138536]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10300
    The total training accuracy:  [147.74107246]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10400
    The total training accuracy:  [147.74079854]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10500
    The total training accuracy:  [147.74050456]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10600
    The total training accuracy:  [147.74020948]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10700
    The total training accuracy:  [147.73990188]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10800
    The total training accuracy:  [147.74140246]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10900
    The total training accuracy:  [147.73931916]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11000
    The total training accuracy:  [147.73903011]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11100
    The total training accuracy:  [147.73872888]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11200
    The total training accuracy:  [147.73841502]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11300
    The total training accuracy:  [147.73813394]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11400
    The total training accuracy:  [147.73784949]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11500
    The total training accuracy:  [147.7375561]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11600
    The total training accuracy:  [147.73725028]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11700
    The total training accuracy:  [147.7483884]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11800
    The total training accuracy:  [147.73666522]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11900
    The total training accuracy:  [147.7363801]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12000
    The total training accuracy:  [147.73608343]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12100
    The total training accuracy:  [147.73577418]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12200
    The total training accuracy:  [147.73611337]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12300
    The total training accuracy:  [147.73519903]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12400
    The total training accuracy:  [147.73491318]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12500
    The total training accuracy:  [147.73461525]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12600
    The total training accuracy:  [147.73430469]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12700
    The total training accuracy:  [147.73454087]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12800
    The total training accuracy:  [147.73373476]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12900
    The total training accuracy:  [147.73344983]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13000
    The total training accuracy:  [147.73315288]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13100
    The total training accuracy:  [147.73284334]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13200
    The total training accuracy:  [147.73265321]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13300
    The total training accuracy:  [147.73227034]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13400
    The total training accuracy:  [147.73198537]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13500
    The total training accuracy:  [147.73168836]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13600
    The total training accuracy:  [147.73137877]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13700
    The total training accuracy:  [147.73183258]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13800
    The total training accuracy:  [147.7308057]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13900
    The total training accuracy:  [147.73051956]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14000
    The total training accuracy:  [147.73022132]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14100
    The total training accuracy:  [147.72991044]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14200
    The total training accuracy:  [147.72988098]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14300
    The total training accuracy:  [147.72934275]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14400
    The total training accuracy:  [147.72905779]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14500
    The total training accuracy:  [147.72876079]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14600
    The total training accuracy:  [147.72845121]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14700
    The total training accuracy:  [147.72887707]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14800
    The total training accuracy:  [147.72787926]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14900
    The total training accuracy:  [147.72759386]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15000
    The total training accuracy:  [147.72729642]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15100
    The total training accuracy:  [147.72698637]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15200
    The total training accuracy:  [147.72675526]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15300
    The total training accuracy:  [147.72641502]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15400
    The total training accuracy:  [147.72612733]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15500
    The total training accuracy:  [147.72582749]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15600
    The total training accuracy:  [147.72551493]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15700
    The total training accuracy:  [147.72525517]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15800
    The total training accuracy:  [147.72495034]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15900
    The total training accuracy:  [147.72465991]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16000
    The total training accuracy:  [147.72435719]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16100
    The total training accuracy:  [147.72454067]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16200
    The total training accuracy:  [147.72376859]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16300
    The total training accuracy:  [147.72348808]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16400
    The total training accuracy:  [147.72319755]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16500
    The total training accuracy:  [147.72289474]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16600
    The total training accuracy:  [147.72548002]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16700
    The total training accuracy:  [147.72230694]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16800
    The total training accuracy:  [147.72202542]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16900
    The total training accuracy:  [147.72173402]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17000
    The total training accuracy:  [147.72143029]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17100
    The total training accuracy:  [147.73306898]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17200
    The total training accuracy:  [147.72084398]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17300
    The total training accuracy:  [147.72056132]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17400
    The total training accuracy:  [147.72026745]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17500
    The total training accuracy:  [147.71996114]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17600
    The total training accuracy:  [147.72246597]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17700
    The total training accuracy:  [147.71938096]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17800
    The total training accuracy:  [147.71909537]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17900
    The total training accuracy:  [147.71879773]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18000
    The total training accuracy:  [147.71848749]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18100
    The total training accuracy:  [147.71833578]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18200
    The total training accuracy:  [147.71791814]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18300
    The total training accuracy:  [147.71762932]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18400
    The total training accuracy:  [147.7173283]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18500
    The total training accuracy:  [147.71701492]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18600
    The total training accuracy:  [147.71674038]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18700
    The total training accuracy:  [147.71645753]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18800
    The total training accuracy:  [147.71616843]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18900
    The total training accuracy:  [147.71586711]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19000
    The total training accuracy:  [147.71556102]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19100
    The total training accuracy:  [147.7152755]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19200
    The total training accuracy:  [147.71499636]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19300
    The total training accuracy:  [147.71470598]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19400
    The total training accuracy:  [147.71440331]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19500
    The total training accuracy:  [147.76227777]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19600
    The total training accuracy:  [147.71381536]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19700
    The total training accuracy:  [147.7135344]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19800
    The total training accuracy:  [147.71324175]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19900
    The total training accuracy:  [147.71293673]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20000
    The total training accuracy:  [147.71292313]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20100
    The total training accuracy:  [147.71235624]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20200
    The total training accuracy:  [147.71207433]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20300
    The total training accuracy:  [147.71178102]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20400
    The total training accuracy:  [147.7114753]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20500
    The total training accuracy:  [147.73065029]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20600
    The total training accuracy:  [147.71089839]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20700
    The total training accuracy:  [147.71062162]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20800
    The total training accuracy:  [147.71033368]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20900
    The total training accuracy:  [147.71003358]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21000
    The total training accuracy:  [147.70972077]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21100
    The total training accuracy:  [147.70946987]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21200
    The total training accuracy:  [147.70916501]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21300
    The total training accuracy:  [147.7088795]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21400
    The total training accuracy:  [147.70858195]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21500
    The total training accuracy:  [147.7082718]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21600
    The total training accuracy:  [147.70818182]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21700
    The total training accuracy:  [147.70770759]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21800
    The total training accuracy:  [147.70742326]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21900
    The total training accuracy:  [147.70712693]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22000
    The total training accuracy:  [147.70681807]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22100
    The total training accuracy:  [147.70676028]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22200
    The total training accuracy:  [147.70624912]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22300
    The total training accuracy:  [147.70596356]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22400
    The total training accuracy:  [147.70566595]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22500
    The total training accuracy:  [147.70535574]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22600
    The total training accuracy:  [147.70520614]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22700
    The total training accuracy:  [147.70479098]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22800
    The total training accuracy:  [147.70450433]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22900
    The total training accuracy:  [147.70420559]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23000
    The total training accuracy:  [147.7038942]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23100
    The total training accuracy:  [147.70361558]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23200
    The total training accuracy:  [147.70333296]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23300
    The total training accuracy:  [147.70304504]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23400
    The total training accuracy:  [147.70274498]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23500
    The total training accuracy:  [147.70243259]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23600
    The total training accuracy:  [147.7021531]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23700
    The total training accuracy:  [147.70187515]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23800
    The total training accuracy:  [147.70158596]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23900
    The total training accuracy:  [147.70128456]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24000
    The total training accuracy:  [147.70266201]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24100
    The total training accuracy:  [147.70069786]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24200
    The total training accuracy:  [147.70041692]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24300
    The total training accuracy:  [147.70012582]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24400
    The total training accuracy:  [147.69982242]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24500
    The total training accuracy:  [147.70175863]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24600
    The total training accuracy:  [147.69923986]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24700
    The total training accuracy:  [147.69895714]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24800
    The total training accuracy:  [147.69866276]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24900
    The total training accuracy:  [147.69835594]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25000
    The total training accuracy:  [147.70028252]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25100
    The total training accuracy:  [147.69778565]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25200
    The total training accuracy:  [147.69750765]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25300
    The total training accuracy:  [147.69721803]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25400
    The total training accuracy:  [147.69691617]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25500
    The total training accuracy:  [147.69794272]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25600
    The total training accuracy:  [147.6963334]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25700
    The total training accuracy:  [147.69605463]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25800
    The total training accuracy:  [147.69576732]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25900
    The total training accuracy:  [147.69546788]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26000
    The total training accuracy:  [147.69515579]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26100
    The total training accuracy:  [147.69487525]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26200
    The total training accuracy:  [147.694601]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26300
    The total training accuracy:  [147.69431526]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26400
    The total training accuracy:  [147.69401747]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26500
    The total training accuracy:  [147.69370709]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26600
    The total training accuracy:  [147.69342311]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26700
    The total training accuracy:  [147.69314621]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26800
    The total training accuracy:  [147.69286011]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26900
    The total training accuracy:  [147.69256195]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27000
    The total training accuracy:  [147.69225118]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27100
    The total training accuracy:  [147.6919901]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27200
    The total training accuracy:  [147.69169057]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27300
    The total training accuracy:  [147.69140258]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27400
    The total training accuracy:  [147.69110244]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27500
    The total training accuracy:  [147.69089627]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27600
    The total training accuracy:  [147.69051987]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27700
    The total training accuracy:  [147.69023361]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27800
    The total training accuracy:  [147.6899423]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27900
    The total training accuracy:  [147.6896387]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28000
    The total training accuracy:  [147.69384378]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28100
    The total training accuracy:  [147.68905907]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28200
    The total training accuracy:  [147.68877683]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28300
    The total training accuracy:  [147.68848279]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28400
    The total training accuracy:  [147.68817634]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28500
    The total training accuracy:  [147.68932855]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28600
    The total training accuracy:  [147.68760574]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28700
    The total training accuracy:  [147.68732319]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28800
    The total training accuracy:  [147.68702879]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28900
    The total training accuracy:  [147.68672196]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29000
    The total training accuracy:  [147.68644282]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29100
    The total training accuracy:  [147.68615198]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29200
    The total training accuracy:  [147.68586761]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29300
    The total training accuracy:  [147.68557127]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29400
    The total training accuracy:  [147.68526241]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29500
    The total training accuracy:  [147.68510232]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29600
    The total training accuracy:  [147.68469732]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29700
    The total training accuracy:  [147.6844097]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29800
    The total training accuracy:  [147.68410996]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29900
    The total training accuracy:  [147.68383582]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30000
    The total training accuracy:  [147.68352749]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30100
    The total training accuracy:  [147.6832429]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30200
    The total training accuracy:  [147.68295281]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30300
    The total training accuracy:  [147.68265049]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30400
    The total training accuracy:  [147.69690123]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30500
    The total training accuracy:  [147.68206949]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30600
    The total training accuracy:  [147.68178675]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30700
    The total training accuracy:  [147.68149285]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30800
    The total training accuracy:  [147.68118654]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30900
    The total training accuracy:  [147.68106852]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31000
    The total training accuracy:  [147.68061711]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31100
    The total training accuracy:  [147.68033458]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31200
    The total training accuracy:  [147.68004015]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31300
    The total training accuracy:  [147.6797333]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31400
    The total training accuracy:  [147.67978862]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31500
    The total training accuracy:  [147.67916508]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31600
    The total training accuracy:  [147.67888076]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31700
    The total training accuracy:  [147.67858447]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31800
    The total training accuracy:  [147.67827567]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31900
    The total training accuracy:  [147.6779879]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32000
    The total training accuracy:  [147.67771249]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32100
    The total training accuracy:  [147.67742551]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32200
    The total training accuracy:  [147.67712644]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32300
    The total training accuracy:  [147.67681519]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32400
    The total training accuracy:  [147.67655125]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32500
    The total training accuracy:  [147.6762623]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32600
    The total training accuracy:  [147.67597559]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32700
    The total training accuracy:  [147.6756768]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32800
    The total training accuracy:  [147.67536566]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32900
    The total training accuracy:  [147.67508779]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33000
    The total training accuracy:  [147.67481196]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33100
    The total training accuracy:  [147.67452492]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33200
    The total training accuracy:  [147.67422579]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33300
    The total training accuracy:  [147.6739267]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33400
    The total training accuracy:  [147.67363839]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33500
    The total training accuracy:  [147.67336048]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33600
    The total training accuracy:  [147.67307151]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33700
    The total training accuracy:  [147.67277036]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33800
    The total training accuracy:  [147.71397686]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33900
    The total training accuracy:  [147.67218849]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34000
    The total training accuracy:  [147.67190725]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34100
    The total training accuracy:  [147.67161476]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34200
    The total training accuracy:  [147.67130994]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34300
    The total training accuracy:  [147.67145337]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34400
    The total training accuracy:  [147.67073689]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34500
    The total training accuracy:  [147.67045175]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34600
    The total training accuracy:  [147.6701546]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34700
    The total training accuracy:  [147.66984494]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34800
    The total training accuracy:  [147.66956261]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34900
    The total training accuracy:  [147.66928374]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35000
    The total training accuracy:  [147.66899325]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35100
    The total training accuracy:  [147.66869052]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35200
    The total training accuracy:  [147.67739324]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35300
    The total training accuracy:  [147.66811374]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35400
    The total training accuracy:  [147.66783205]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35500
    The total training accuracy:  [147.66753886]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35600
    The total training accuracy:  [147.6672333]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35700
    The total training accuracy:  [147.66742951]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35800
    The total training accuracy:  [147.66666486]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35900
    The total training accuracy:  [147.66638246]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36000
    The total training accuracy:  [147.66608818]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36100
    The total training accuracy:  [147.66578149]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36200
    The total training accuracy:  [147.66555355]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36300
    The total training accuracy:  [147.66521616]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36400
    The total training accuracy:  [147.66493206]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36500
    The total training accuracy:  [147.664636]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36600
    The total training accuracy:  [147.66432747]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36700
    The total training accuracy:  [147.66412282]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36800
    The total training accuracy:  [147.66376803]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36900
    The total training accuracy:  [147.66348289]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37000
    The total training accuracy:  [147.66318576]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37100
    The total training accuracy:  [147.66287609]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37200
    The total training accuracy:  [147.66260275]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37300
    The total training accuracy:  [147.66231945]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37400
    The total training accuracy:  [147.66203242]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37500
    The total training accuracy:  [147.66173332]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37600
    The total training accuracy:  [147.66288939]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37700
    The total training accuracy:  [147.66115213]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37800
    The total training accuracy:  [147.66086941]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37900
    The total training accuracy:  [147.66057901]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38000
    The total training accuracy:  [147.66027638]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38100
    The total training accuracy:  [147.66534625]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38200
    The total training accuracy:  [147.65970056]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38300
    The total training accuracy:  [147.6594181]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38400
    The total training accuracy:  [147.6591238]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38500
    The total training accuracy:  [147.6588171]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38600
    The total training accuracy:  [147.65853084]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38700
    The total training accuracy:  [147.65825223]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38800
    The total training accuracy:  [147.65796625]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38900
    The total training accuracy:  [147.65766823]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39000
    The total training accuracy:  [147.65737616]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39100
    The total training accuracy:  [147.65708631]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39200
    The total training accuracy:  [147.65680281]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39300
    The total training accuracy:  [147.65651286]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39400
    The total training accuracy:  [147.6562107]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39500
    The total training accuracy:  [147.66249176]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39600
    The total training accuracy:  [147.65563439]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39700
    The total training accuracy:  [147.6553512]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39800
    The total training accuracy:  [147.65505632]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39900
    The total training accuracy:  [147.65474902]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40000
    The total training accuracy:  [147.6545464]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40100
    The total training accuracy:  [147.65418655]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40200
    The total training accuracy:  [147.6538999]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40300
    The total training accuracy:  [147.65360119]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40400
    The total training accuracy:  [147.65349799]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40500
    The total training accuracy:  [147.65301941]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40600
    The total training accuracy:  [147.6527405]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40700
    The total training accuracy:  [147.65245255]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40800
    The total training accuracy:  [147.65215248]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40900
    The total training accuracy:  [147.65239204]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41000
    The total training accuracy:  [147.65157227]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41100
    The total training accuracy:  [147.65129344]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41200
    The total training accuracy:  [147.65100308]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41300
    The total training accuracy:  [147.65070049]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41400
    The total training accuracy:  [147.65524055]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41500
    The total training accuracy:  [147.65012676]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41600
    The total training accuracy:  [147.64984453]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41700
    The total training accuracy:  [147.64955046]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41800
    The total training accuracy:  [147.64924401]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41900
    The total training accuracy:  [147.64914834]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42000
    The total training accuracy:  [147.64868085]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42100
    The total training accuracy:  [147.64839538]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42200
    The total training accuracy:  [147.64809793]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42300
    The total training accuracy:  [147.64779214]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42400
    The total training accuracy:  [147.64752218]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42500
    The total training accuracy:  [147.64723472]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42600
    The total training accuracy:  [147.64694636]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42700
    The total training accuracy:  [147.64664588]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42800
    The total training accuracy:  [147.65682332]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42900
    The total training accuracy:  [147.64606817]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43000
    The total training accuracy:  [147.64578686]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43100
    The total training accuracy:  [147.64549445]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43200
    The total training accuracy:  [147.64518972]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43300
    The total training accuracy:  [147.64513906]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43400
    The total training accuracy:  [147.64462198]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43500
    The total training accuracy:  [147.64433649]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43600
    The total training accuracy:  [147.64403902]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43700
    The total training accuracy:  [147.64385271]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43800
    The total training accuracy:  [147.64345389]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43900
    The total training accuracy:  [147.64317449]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44000
    The total training accuracy:  [147.64288381]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44100
    The total training accuracy:  [147.64258092]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44200
    The total training accuracy:  [147.6438897]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44300
    The total training accuracy:  [147.6420092]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44400
    The total training accuracy:  [147.64172542]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44500
    The total training accuracy:  [147.64142973]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44600
    The total training accuracy:  [147.64112161]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44700
    The total training accuracy:  [147.64084626]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44800
    The total training accuracy:  [147.64056308]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44900
    The total training accuracy:  [147.64027423]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45000
    The total training accuracy:  [147.63997324]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45100
    The total training accuracy:  [147.64078052]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45200
    The total training accuracy:  [147.63939711]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45300
    The total training accuracy:  [147.6391145]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45400
    The total training accuracy:  [147.63882003]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45500
    The total training accuracy:  [147.63851317]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45600
    The total training accuracy:  [147.6382484]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45700
    The total training accuracy:  [147.63795129]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45800
    The total training accuracy:  [147.63766269]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45900
    The total training accuracy:  [147.63736197]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46000
    The total training accuracy:  [147.63899382]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46100
    The total training accuracy:  [147.63678563]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46200
    The total training accuracy:  [147.63650227]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46300
    The total training accuracy:  [147.63620722]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46400
    The total training accuracy:  [147.63589975]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46500
    The total training accuracy:  [147.63568301]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46600
    The total training accuracy:  [147.63534401]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46700
    The total training accuracy:  [147.63505991]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46800
    The total training accuracy:  [147.63476387]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46900
    The total training accuracy:  [147.63445542]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47000
    The total training accuracy:  [147.63418348]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47100
    The total training accuracy:  [147.63390161]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47200
    The total training accuracy:  [147.63361512]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47300
    The total training accuracy:  [147.63331659]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47400
    The total training accuracy:  [147.71973705]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47500
    The total training accuracy:  [147.63273616]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47600
    The total training accuracy:  [147.63245765]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47700
    The total training accuracy:  [147.63216746]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47800
    The total training accuracy:  [147.63186508]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47900
    The total training accuracy:  [147.6342581]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48000
    The total training accuracy:  [147.63129561]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48100
    The total training accuracy:  [147.63101485]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48200
    The total training accuracy:  [147.63072238]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48300
    The total training accuracy:  [147.63041761]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48400
    The total training accuracy:  [147.63042511]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48500
    The total training accuracy:  [147.62985498]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48600
    The total training accuracy:  [147.62957231]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48700
    The total training accuracy:  [147.62927779]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48800
    The total training accuracy:  [147.62897088]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48900
    The total training accuracy:  [147.62873941]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49000
    The total training accuracy:  [147.62841345]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49100
    The total training accuracy:  [147.62812761]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49200
    The total training accuracy:  [147.62782977]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49300
    The total training accuracy:  [147.62754347]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49400
    The total training accuracy:  [147.62725784]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49500
    The total training accuracy:  [147.62697731]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49600
    The total training accuracy:  [147.62669441]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49700
    The total training accuracy:  [147.62639966]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49800
    The total training accuracy:  [147.6260925]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49900
    The total training accuracy:  [147.62588091]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50000
    The total training accuracy:  [147.6255389]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50100
    The total training accuracy:  [147.62525613]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50200
    The total training accuracy:  [147.62496151]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50300
    The total training accuracy:  [147.6246545]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50400
    The total training accuracy:  [147.62440353]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50500
    The total training accuracy:  [147.62409951]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50600
    The total training accuracy:  [147.62381514]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50700
    The total training accuracy:  [147.62351884]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50800
    The total training accuracy:  [147.62321049]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50900
    The total training accuracy:  [147.62293895]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51000
    The total training accuracy:  [147.62266126]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51100
    The total training accuracy:  [147.62237645]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51200
    The total training accuracy:  [147.62207969]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51300
    The total training accuracy:  [147.62181627]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51400
    The total training accuracy:  [147.62150477]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51500
    The total training accuracy:  [147.62122163]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51600
    The total training accuracy:  [147.62093459]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51700
    The total training accuracy:  [147.6206355]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51800
    The total training accuracy:  [147.63177407]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51900
    The total training accuracy:  [147.62005982]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52000
    The total training accuracy:  [147.61978247]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52100
    The total training accuracy:  [147.61949369]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52200
    The total training accuracy:  [147.61919279]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52300
    The total training accuracy:  [147.6267661]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52400
    The total training accuracy:  [147.61862249]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52500
    The total training accuracy:  [147.61834388]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52600
    The total training accuracy:  [147.61805376]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52700
    The total training accuracy:  [147.61775145]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52800
    The total training accuracy:  [147.61745732]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52900
    The total training accuracy:  [147.61718467]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53000
    The total training accuracy:  [147.6169036]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53100
    The total training accuracy:  [147.6166108]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53200
    The total training accuracy:  [147.61630569]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53300
    The total training accuracy:  [147.6161407]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53400
    The total training accuracy:  [147.61574729]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53500
    The total training accuracy:  [147.61546464]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53600
    The total training accuracy:  [147.61517015]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53700
    The total training accuracy:  [147.61486328]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53800
    The total training accuracy:  [147.61458476]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53900
    The total training accuracy:  [147.61431208]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54000
    The total training accuracy:  [147.61403049]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54100
    The total training accuracy:  [147.6137371]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54200
    The total training accuracy:  [147.61343139]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54300
    The total training accuracy:  [147.61317238]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54400
    The total training accuracy:  [147.61287562]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54500
    The total training accuracy:  [147.61259287]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54600
    The total training accuracy:  [147.61229827]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54700
    The total training accuracy:  [147.6119913]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54800
    The total training accuracy:  [147.61176561]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54900
    The total training accuracy:  [147.61144118]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55000
    The total training accuracy:  [147.61115965]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55100
    The total training accuracy:  [147.61086632]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55200
    The total training accuracy:  [147.61056067]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55300
    The total training accuracy:  [147.61028872]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55400
    The total training accuracy:  [147.61000605]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55500
    The total training accuracy:  [147.60972436]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55600
    The total training accuracy:  [147.60943088]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55700
    The total training accuracy:  [147.60912507]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55800
    The total training accuracy:  [147.60889898]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55900
    The total training accuracy:  [147.60857344]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56000
    The total training accuracy:  [147.60829504]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56100
    The total training accuracy:  [147.60800499]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56200
    The total training accuracy:  [147.60770277]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56300
    The total training accuracy:  [147.60998833]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56400
    The total training accuracy:  [147.60714009]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56500
    The total training accuracy:  [147.60686415]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56600
    The total training accuracy:  [147.60657674]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56700
    The total training accuracy:  [147.60627727]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56800
    The total training accuracy:  [147.61534027]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56900
    The total training accuracy:  [147.60570612]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57000
    The total training accuracy:  [147.60543064]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57100
    The total training accuracy:  [147.6051437]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57200
    The total training accuracy:  [147.60484472]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57300
    The total training accuracy:  [147.61050795]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57400
    The total training accuracy:  [147.60427266]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57500
    The total training accuracy:  [147.60399571]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57600
    The total training accuracy:  [147.60370772]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57700
    The total training accuracy:  [147.60340765]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57800
    The total training accuracy:  [147.60992735]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57900
    The total training accuracy:  [147.60283829]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58000
    The total training accuracy:  [147.60255954]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58100
    The total training accuracy:  [147.60226936]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58200
    The total training accuracy:  [147.60196701]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58300
    The total training accuracy:  [147.60300993]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58400
    The total training accuracy:  [147.6014073]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58500
    The total training accuracy:  [147.60113559]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58600
    The total training accuracy:  [147.60085252]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58700
    The total training accuracy:  [147.6005576]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58800
    The total training accuracy:  [147.60025031]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58900
    The total training accuracy:  [147.59997548]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59000
    The total training accuracy:  [147.59970463]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59100
    The total training accuracy:  [147.59942385]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59200
    The total training accuracy:  [147.59913131]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59300
    The total training accuracy:  [147.5988265]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59400
    The total training accuracy:  [147.5985861]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59500
    The total training accuracy:  [147.59827315]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59600
    The total training accuracy:  [147.59799392]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59700
    The total training accuracy:  [147.597703]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59800
    The total training accuracy:  [147.59739989]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59900
    The total training accuracy:  [147.59743706]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60000
    The total training accuracy:  [147.59684061]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60100
    The total training accuracy:  [147.59656061]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60200
    The total training accuracy:  [147.59626891]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60300
    The total training accuracy:  [147.59596497]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60400
    The total training accuracy:  [147.59583118]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60500
    The total training accuracy:  [147.59540788]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60600
    The total training accuracy:  [147.59512645]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60700
    The total training accuracy:  [147.59483325]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60800
    The total training accuracy:  [147.59452775]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60900
    The total training accuracy:  [147.59425363]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61000
    The total training accuracy:  [147.59397408]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61100
    The total training accuracy:  [147.59368969]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61200
    The total training accuracy:  [147.5933934]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61300
    The total training accuracy:  [147.62566114]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61400
    The total training accuracy:  [147.59281743]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61500
    The total training accuracy:  [147.59253873]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61600
    The total training accuracy:  [147.59225032]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61700
    The total training accuracy:  [147.59194982]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61800
    The total training accuracy:  [147.59344773]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61900
    The total training accuracy:  [147.5913827]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62000
    The total training accuracy:  [147.59110118]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62100
    The total training accuracy:  [147.5908079]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62200
    The total training accuracy:  [147.59050233]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62300
    The total training accuracy:  [147.59025744]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62400
    The total training accuracy:  [147.58994992]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62500
    The total training accuracy:  [147.58966572]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62600
    The total training accuracy:  [147.58936962]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62700
    The total training accuracy:  [147.59299198]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62800
    The total training accuracy:  [147.58879282]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62900
    The total training accuracy:  [147.58851783]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63000
    The total training accuracy:  [147.58823187]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63100
    The total training accuracy:  [147.58793394]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63200
    The total training accuracy:  [147.60349469]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63300
    The total training accuracy:  [147.58736198]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63400
    The total training accuracy:  [147.58708514]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63500
    The total training accuracy:  [147.58679688]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63600
    The total training accuracy:  [147.58649655]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63700
    The total training accuracy:  [147.58628463]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63800
    The total training accuracy:  [147.58593105]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63900
    The total training accuracy:  [147.58565144]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64000
    The total training accuracy:  [147.58536023]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64100
    The total training accuracy:  [147.58505681]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64200
    The total training accuracy:  [147.58481541]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64300
    The total training accuracy:  [147.58449888]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64400
    The total training accuracy:  [147.58421534]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64500
    The total training accuracy:  [147.58391994]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64600
    The total training accuracy:  [147.58713183]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64700
    The total training accuracy:  [147.58334544]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64800
    The total training accuracy:  [147.58306538]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64900
    The total training accuracy:  [147.58277749]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65000
    The total training accuracy:  [147.58247754]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65100
    The total training accuracy:  [147.58254323]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65200
    The total training accuracy:  [147.58191097]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65300
    The total training accuracy:  [147.58163003]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65400
    The total training accuracy:  [147.58133734]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65500
    The total training accuracy:  [147.58103239]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65600
    The total training accuracy:  [147.58082735]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65700
    The total training accuracy:  [147.58048291]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65800
    The total training accuracy:  [147.58020271]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65900
    The total training accuracy:  [147.57991081]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66000
    The total training accuracy:  [147.57960668]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66100
    The total training accuracy:  [147.57934043]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66200
    The total training accuracy:  [147.57905493]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66300
    The total training accuracy:  [147.57877522]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66400
    The total training accuracy:  [147.57848382]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66500
    The total training accuracy:  [147.57818021]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66600
    The total training accuracy:  [147.5779954]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66700
    The total training accuracy:  [147.57762623]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66800
    The total training accuracy:  [147.5773454]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66900
    The total training accuracy:  [147.57705284]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67000
    The total training accuracy:  [147.57674803]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67100
    The total training accuracy:  [147.57655016]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67200
    The total training accuracy:  [147.57619924]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67300
    The total training accuracy:  [147.5759194]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67400
    The total training accuracy:  [147.57562787]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67500
    The total training accuracy:  [147.57532412]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67600
    The total training accuracy:  [147.57506297]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67700
    The total training accuracy:  [147.57477101]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67800
    The total training accuracy:  [147.57448991]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67900
    The total training accuracy:  [147.57419706]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68000
    The total training accuracy:  [147.57389195]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68100
    The total training accuracy:  [147.57361556]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68200
    The total training accuracy:  [147.5733422]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68300
    The total training accuracy:  [147.57305894]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68400
    The total training accuracy:  [147.57276384]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68500
    The total training accuracy:  [147.57254267]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68600
    The total training accuracy:  [147.57219342]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68700
    The total training accuracy:  [147.57191595]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68800
    The total training accuracy:  [147.57163327]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68900
    The total training accuracy:  [147.57133877]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69000
    The total training accuracy:  [147.57106232]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69100
    The total training accuracy:  [147.5707692]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69200
    The total training accuracy:  [147.57048809]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69300
    The total training accuracy:  [147.57020394]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69400
    The total training accuracy:  [147.5699079]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69500
    The total training accuracy:  [147.57883921]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69600
    The total training accuracy:  [147.56933496]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69700
    The total training accuracy:  [147.56905885]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69800
    The total training accuracy:  [147.56877185]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69900
    The total training accuracy:  [147.56847284]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70000
    The total training accuracy:  [147.56987886]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70100
    The total training accuracy:  [147.56790721]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70200
    The total training accuracy:  [147.56762792]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70300
    The total training accuracy:  [147.5673371]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70400
    The total training accuracy:  [147.5670341]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70500
    The total training accuracy:  [147.56687918]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70600
    The total training accuracy:  [147.56647934]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70700
    The total training accuracy:  [147.56619702]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70800
    The total training accuracy:  [147.56590291]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70900
    The total training accuracy:  [147.56559989]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71000
    The total training accuracy:  [147.56532515]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71100
    The total training accuracy:  [147.56505387]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71200
    The total training accuracy:  [147.5647714]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71300
    The total training accuracy:  [147.56447714]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71400
    The total training accuracy:  [147.56417263]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71500
    The total training accuracy:  [147.56390627]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71600
    The total training accuracy:  [147.56363016]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71700
    The total training accuracy:  [147.5633492]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71800
    The total training accuracy:  [147.5630565]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71900
    The total training accuracy:  [147.56275156]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72000
    The total training accuracy:  [147.56250531]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72100
    The total training accuracy:  [147.56220474]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72200
    The total training accuracy:  [147.56192293]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72300
    The total training accuracy:  [147.56162936]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72400
    The total training accuracy:  [147.56132908]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72500
    The total training accuracy:  [147.56106033]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72600
    The total training accuracy:  [147.56077796]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72700
    The total training accuracy:  [147.56049357]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72800
    The total training accuracy:  [147.56019731]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72900
    The total training accuracy:  [147.5670985]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73000
    The total training accuracy:  [147.55962651]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73100
    The total training accuracy:  [147.5593514]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73200
    The total training accuracy:  [147.55906486]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73300
    The total training accuracy:  [147.55876634]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73400
    The total training accuracy:  [147.59012387]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73500
    The total training accuracy:  [147.55820427]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73600
    The total training accuracy:  [147.55793835]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73700
    The total training accuracy:  [147.55766156]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73800
    The total training accuracy:  [147.55737322]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73900
    The total training accuracy:  [147.55707283]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74000
    The total training accuracy:  [147.55687862]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74100
    The total training accuracy:  [147.55651641]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74200
    The total training accuracy:  [147.55624341]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74300
    The total training accuracy:  [147.5559591]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74400
    The total training accuracy:  [147.55566291]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74500
    The total training accuracy:  [147.5592719]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74600
    The total training accuracy:  [147.55509339]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74700
    The total training accuracy:  [147.55482147]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74800
    The total training accuracy:  [147.55453836]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74900
    The total training accuracy:  [147.55424342]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75000
    The total training accuracy:  [147.56099615]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75100
    The total training accuracy:  [147.55367224]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75200
    The total training accuracy:  [147.55339855]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75300
    The total training accuracy:  [147.55311546]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75400
    The total training accuracy:  [147.55282055]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75500
    The total training accuracy:  [147.59084181]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75600
    The total training accuracy:  [147.5522476]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75700
    The total training accuracy:  [147.55197429]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75800
    The total training accuracy:  [147.55168961]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75900
    The total training accuracy:  [147.55139305]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76000
    The total training accuracy:  [147.5511282]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76100
    The total training accuracy:  [147.55082546]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76200
    The total training accuracy:  [147.55055016]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76300
    The total training accuracy:  [147.55026394]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76400
    The total training accuracy:  [147.54996578]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76500
    The total training accuracy:  [147.55183936]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76600
    The total training accuracy:  [147.54940225]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76700
    The total training accuracy:  [147.54912519]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76800
    The total training accuracy:  [147.54883673]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76900
    The total training accuracy:  [147.54853622]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77000
    The total training accuracy:  [147.54843956]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77100
    The total training accuracy:  [147.54797931]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77200
    The total training accuracy:  [147.54770049]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77300
    The total training accuracy:  [147.54741005]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77400
    The total training accuracy:  [147.54710747]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77500
    The total training accuracy:  [147.5469106]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77600
    The total training accuracy:  [147.54655549]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77700
    The total training accuracy:  [147.54627342]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77800
    The total training accuracy:  [147.54597959]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77900
    The total training accuracy:  [147.54905078]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78000
    The total training accuracy:  [147.54540868]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78100
    The total training accuracy:  [147.54513061]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78200
    The total training accuracy:  [147.54484488]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78300
    The total training accuracy:  [147.54454722]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78400
    The total training accuracy:  [147.54475018]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78500
    The total training accuracy:  [147.54398433]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78600
    The total training accuracy:  [147.5437113]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78700
    The total training accuracy:  [147.54342691]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78800
    The total training accuracy:  [147.54313065]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78900
    The total training accuracy:  [147.56922477]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79000
    The total training accuracy:  [147.54256461]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79100
    The total training accuracy:  [147.54229166]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79200
    The total training accuracy:  [147.54200817]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79300
    The total training accuracy:  [147.54171285]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79400
    The total training accuracy:  [147.54204602]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79500
    The total training accuracy:  [147.54114485]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79600
    The total training accuracy:  [147.54087128]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79700
    The total training accuracy:  [147.54058778]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79800
    The total training accuracy:  [147.54029245]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79900
    The total training accuracy:  [147.57478805]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80000
    The total training accuracy:  [147.53972323]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80100
    The total training accuracy:  [147.53944913]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80200
    The total training accuracy:  [147.53916375]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80300
    The total training accuracy:  [147.53886645]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80400
    The total training accuracy:  [147.54162591]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80500
    The total training accuracy:  [147.53830232]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80600
    The total training accuracy:  [147.53802505]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80700
    The total training accuracy:  [147.5377363]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80800
    The total training accuracy:  [147.5374355]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80900
    The total training accuracy:  [147.53716258]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81000
    The total training accuracy:  [147.53688025]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81100
    The total training accuracy:  [147.53659907]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81200
    The total training accuracy:  [147.53630616]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81300
    The total training accuracy:  [147.53627865]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81400
    The total training accuracy:  [147.53573612]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81500
    The total training accuracy:  [147.53545702]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81600
    The total training accuracy:  [147.5351717]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81700
    The total training accuracy:  [147.53487448]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81800
    The total training accuracy:  [147.54014267]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81900
    The total training accuracy:  [147.53431178]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82000
    The total training accuracy:  [147.53403722]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82100
    The total training accuracy:  [147.53375124]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82200
    The total training accuracy:  [147.53345333]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82300
    The total training accuracy:  [147.53411736]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82400
    The total training accuracy:  [147.53289421]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82500
    The total training accuracy:  [147.53262241]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82600
    The total training accuracy:  [147.53233953]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82700
    The total training accuracy:  [147.53204487]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82800
    The total training accuracy:  [147.62863934]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82900
    The total training accuracy:  [147.53147592]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83000
    The total training accuracy:  [147.53120467]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83100
    The total training accuracy:  [147.53092243]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83200
    The total training accuracy:  [147.53062843]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83300
    The total training accuracy:  [147.57398189]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83400
    The total training accuracy:  [147.53005753]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83500
    The total training accuracy:  [147.52978506]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83600
    The total training accuracy:  [147.52950135]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83700
    The total training accuracy:  [147.52920582]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83800
    The total training accuracy:  [147.54166115]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83900
    The total training accuracy:  [147.52863925]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84000
    The total training accuracy:  [147.52836379]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84100
    The total training accuracy:  [147.52807715]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84200
    The total training accuracy:  [147.52777855]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84300
    The total training accuracy:  [147.52795024]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84400
    The total training accuracy:  [147.52721951]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84500
    The total training accuracy:  [147.52694039]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84600
    The total training accuracy:  [147.52664965]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84700
    The total training accuracy:  [147.5263468]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84800
    The total training accuracy:  [147.52607293]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84900
    The total training accuracy:  [147.52579867]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85000
    The total training accuracy:  [147.52551525]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85100
    The total training accuracy:  [147.52522001]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85200
    The total training accuracy:  [147.53550114]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85300
    The total training accuracy:  [147.5246526]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85400
    The total training accuracy:  [147.52437571]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85500
    The total training accuracy:  [147.52408729]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85600
    The total training accuracy:  [147.52378685]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85700
    The total training accuracy:  [147.52359332]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85800
    The total training accuracy:  [147.52323217]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85900
    The total training accuracy:  [147.52294978]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86000
    The total training accuracy:  [147.52265563]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86100
    The total training accuracy:  [147.53015497]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86200
    The total training accuracy:  [147.52208629]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86300
    The total training accuracy:  [147.52180983]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86400
    The total training accuracy:  [147.52152222]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86500
    The total training accuracy:  [147.52122262]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86600
    The total training accuracy:  [147.52132209]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86700
    The total training accuracy:  [147.52067012]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86800
    The total training accuracy:  [147.52039475]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86900
    The total training accuracy:  [147.52010795]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87000
    The total training accuracy:  [147.5198092]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87100
    The total training accuracy:  [147.51991681]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87200
    The total training accuracy:  [147.51925568]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87300
    The total training accuracy:  [147.51898396]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87400
    The total training accuracy:  [147.518701]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87500
    The total training accuracy:  [147.51840625]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87600
    The total training accuracy:  [147.56510617]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87700
    The total training accuracy:  [147.51784148]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87800
    The total training accuracy:  [147.51756952]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87900
    The total training accuracy:  [147.51728768]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88000
    The total training accuracy:  [147.51699412]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88100
    The total training accuracy:  [147.57133717]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88200
    The total training accuracy:  [147.51642643]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88300
    The total training accuracy:  [147.5161532]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88400
    The total training accuracy:  [147.51587046]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88500
    The total training accuracy:  [147.51557595]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88600
    The total training accuracy:  [147.53688329]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88700
    The total training accuracy:  [147.51500986]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88800
    The total training accuracy:  [147.51473511]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88900
    The total training accuracy:  [147.51444974]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89000
    The total training accuracy:  [147.51415248]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89100
    The total training accuracy:  [147.51759612]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89200
    The total training accuracy:  [147.51359423]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89300
    The total training accuracy:  [147.51332007]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89400
    The total training accuracy:  [147.51303455]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89500
    The total training accuracy:  [147.51273714]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89600
    The total training accuracy:  [147.51638199]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89700
    The total training accuracy:  [147.51218031]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89800
    The total training accuracy:  [147.51190788]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89900
    The total training accuracy:  [147.5116242]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90000
    The total training accuracy:  [147.51132871]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90100
    The total training accuracy:  [147.53110917]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90200
    The total training accuracy:  [147.51076679]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90300
    The total training accuracy:  [147.51049496]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90400
    The total training accuracy:  [147.51021245]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90500
    The total training accuracy:  [147.50991818]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90600
    The total training accuracy:  [147.55761967]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90700
    The total training accuracy:  [147.50935286]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90800
    The total training accuracy:  [147.5090809]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90900
    The total training accuracy:  [147.50879834]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91000
    The total training accuracy:  [147.50850403]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91100
    The total training accuracy:  [147.50838162]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91200
    The total training accuracy:  [147.50793933]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91300
    The total training accuracy:  [147.50766602]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91400
    The total training accuracy:  [147.50738244]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91500
    The total training accuracy:  [147.50708705]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91600
    The total training accuracy:  [147.5142583]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91700
    The total training accuracy:  [147.50652448]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91800
    The total training accuracy:  [147.50625011]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91900
    The total training accuracy:  [147.50596451]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92000
    The total training accuracy:  [147.50566702]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92100
    The total training accuracy:  [147.50679512]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92200
    The total training accuracy:  [147.50510994]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92300
    The total training accuracy:  [147.50483306]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92400
    The total training accuracy:  [147.50454469]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92500
    The total training accuracy:  [147.5042443]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92600
    The total training accuracy:  [147.50401446]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92700
    The total training accuracy:  [147.50369514]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92800
    The total training accuracy:  [147.50341556]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92900
    The total training accuracy:  [147.50312435]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93000
    The total training accuracy:  [147.50283745]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93100
    The total training accuracy:  [147.50255246]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93200
    The total training accuracy:  [147.50227965]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93300
    The total training accuracy:  [147.50199699]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93400
    The total training accuracy:  [147.50170257]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93500
    The total training accuracy:  [147.50460181]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93600
    The total training accuracy:  [147.50113886]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93700
    The total training accuracy:  [147.50086579]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93800
    The total training accuracy:  [147.50058186]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93900
    The total training accuracy:  [147.50028612]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94000
    The total training accuracy:  [147.50014704]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94100
    The total training accuracy:  [147.49972567]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94200
    The total training accuracy:  [147.49945117]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94300
    The total training accuracy:  [147.49916527]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94400
    The total training accuracy:  [147.49886747]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94500
    The total training accuracy:  [147.49863374]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94600
    The total training accuracy:  [147.49831228]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94700
    The total training accuracy:  [147.49803476]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94800
    The total training accuracy:  [147.49774571]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94900
    The total training accuracy:  [147.49744463]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95000
    The total training accuracy:  [147.49723041]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95100
    The total training accuracy:  [147.49689965]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95200
    The total training accuracy:  [147.49662075]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95300
    The total training accuracy:  [147.49633026]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95400
    The total training accuracy:  [147.49602878]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95500
    The total training accuracy:  [147.49575687]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95600
    The total training accuracy:  [147.49548579]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95700
    The total training accuracy:  [147.49520403]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95800
    The total training accuracy:  [147.49491056]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95900
    The total training accuracy:  [147.51943108]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96000
    The total training accuracy:  [147.49434455]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96100
    The total training accuracy:  [147.49407008]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96200
    The total training accuracy:  [147.49378426]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96300
    The total training accuracy:  [147.49348654]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96400
    The total training accuracy:  [147.49377153]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96500
    The total training accuracy:  [147.49293178]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96600
    The total training accuracy:  [147.49265372]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96700
    The total training accuracy:  [147.49236411]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96800
    The total training accuracy:  [147.49206246]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96900
    The total training accuracy:  [147.49181302]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97000
    The total training accuracy:  [147.49151863]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97100
    The total training accuracy:  [147.49123745]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97200
    The total training accuracy:  [147.49094458]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97300
    The total training accuracy:  [147.57327215]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97400
    The total training accuracy:  [147.49037827]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97500
    The total training accuracy:  [147.4901078]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97600
    The total training accuracy:  [147.48982614]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97700
    The total training accuracy:  [147.48953276]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97800
    The total training accuracy:  [147.48937368]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97900
    The total training accuracy:  [147.48896841]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98000
    The total training accuracy:  [147.48869565]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98100
    The total training accuracy:  [147.48841209]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98200
    The total training accuracy:  [147.48811674]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98300
    The total training accuracy:  [147.49080888]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98400
    The total training accuracy:  [147.48755705]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98500
    The total training accuracy:  [147.48728156]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98600
    The total training accuracy:  [147.48699467]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98700
    The total training accuracy:  [147.48669584]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98800
    The total training accuracy:  [147.48656205]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98900
    The total training accuracy:  [147.48614498]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99000
    The total training accuracy:  [147.48586558]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99100
    The total training accuracy:  [147.48557459]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99200
    The total training accuracy:  [147.4861822]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99300
    The total training accuracy:  [147.48500392]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99400
    The total training accuracy:  [147.48473159]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99500
    The total training accuracy:  [147.48444799]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99600
    The total training accuracy:  [147.4841526]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99700
    The total training accuracy:  [147.48544866]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99800
    The total training accuracy:  [147.48359369]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99900
    The total training accuracy:  [147.48331776]
    ---------------------------------------------------------------------------------------------------
    Trial number: 100000
    The total training accuracy:  [147.48303049]
    ---------------------------------------------------------------------------------------------------
    


```python
pow_vals = []

v_range = np.linspace(3.5,13.5,100)
dens_range = np.linspace(.9, 1.3,100)

X, Y = np.meshgrid(v_range, dens_range)

dv = (13-4)/300
for v in v_range:
    part = []
    for d in dens_range:
        t = []
        part.append(deg6_2mw_polynomial_model.predict(polynomial_transform(np.array([d,v],ndmin=2).T,6,t))[0,0])
    pow_vals.append(part)

Z = np.asarray(pow_vals)

```


```python
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=1)
ax.view_init(30, 10)
plt.show()
```


    
![png](output_51_0.png)
    


##### A Polynomial Model of the Power Output v.s. Wind Speed curve at constant Air density (1.225 kg/m3) : 


```python
power_out_data = vestas_2mw_data.loc[:,1.225]

X_train = []
X_train = np.asarray(polynomial_transform(power_out_data.index.values.reshape(1, power_out_data.size), 5, X_train))

Y_train = power_out_data.values.reshape(1, power_out_data.size)

vestas_2mw_std_model = NN(1,[X_train.shape[0], 1], ["None", ], losses.mse_loss, losses.mse_loss_deriv,AdaM_opt=True)
```


```python
# Hyperparameter tuning:
vestas_2mw_std_model.reglr_type = "L1"
vestas_2mw_std_model.lmbd = 0

# vestas_2mw_std_model.frwrd_trials = 10000
# vestas_2mw_std_model.back_trials = 10000

# Manipulating curve manually:
# vestas_2mw_std_model.b_layers[1][0,0] += 1

# Training:
total_trials = 100001
batch_size = X_train.shape[1] // 1

for trial in range(total_trials):

    # Hyperparameter tuning:
    # vestas_2mw_std_model.alpha = 0.0001
    vestas_2mw_std_model.alpha = (1 / (trial+1))

    # Stochastic GD 'mask' :
    mask = np.random.choice(X_train.shape[1], batch_size, replace=False)

    train_loss = (NN.training_cycle(vestas_2mw_std_model, X=X_train[:,mask], Y=Y_train[:,mask]))
    if((trial)%500==0):
        print("Trial number:",trial)
        print("The total training accuracy: ",train_loss)
        print("---------------------------------------------------------------------------------------------------")
```

    Trial number: 0
    The total training accuracy:  [377.84234466]
    ---------------------------------------------------------------------------------------------------
    Trial number: 500
    The total training accuracy:  [376.31497988]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1000
    The total training accuracy:  [376.31496272]
    ---------------------------------------------------------------------------------------------------
    Trial number: 1500
    The total training accuracy:  [376.31494975]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2000
    The total training accuracy:  [376.31493791]
    ---------------------------------------------------------------------------------------------------
    Trial number: 2500
    The total training accuracy:  [376.31492609]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3000
    The total training accuracy:  [376.31491369]
    ---------------------------------------------------------------------------------------------------
    Trial number: 3500
    The total training accuracy:  [376.31490021]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4000
    The total training accuracy:  [376.31488522]
    ---------------------------------------------------------------------------------------------------
    Trial number: 4500
    The total training accuracy:  [376.31486823]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5000
    The total training accuracy:  [376.31484872]
    ---------------------------------------------------------------------------------------------------
    Trial number: 5500
    The total training accuracy:  [376.31482605]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6000
    The total training accuracy:  [376.31479948]
    ---------------------------------------------------------------------------------------------------
    Trial number: 6500
    The total training accuracy:  [376.31476809]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7000
    The total training accuracy:  [376.31473078]
    ---------------------------------------------------------------------------------------------------
    Trial number: 7500
    The total training accuracy:  [376.3146862]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8000
    The total training accuracy:  [376.31463267]
    ---------------------------------------------------------------------------------------------------
    Trial number: 8500
    The total training accuracy:  [376.31456812]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9000
    The total training accuracy:  [376.31449004]
    ---------------------------------------------------------------------------------------------------
    Trial number: 9500
    The total training accuracy:  [376.31439526]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10000
    The total training accuracy:  [376.31427992]
    ---------------------------------------------------------------------------------------------------
    Trial number: 10500
    The total training accuracy:  [376.31413919]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11000
    The total training accuracy:  [376.31396715]
    ---------------------------------------------------------------------------------------------------
    Trial number: 11500
    The total training accuracy:  [376.31375641]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12000
    The total training accuracy:  [376.31349785]
    ---------------------------------------------------------------------------------------------------
    Trial number: 12500
    The total training accuracy:  [376.31318019]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13000
    The total training accuracy:  [376.31278947]
    ---------------------------------------------------------------------------------------------------
    Trial number: 13500
    The total training accuracy:  [376.31230848]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14000
    The total training accuracy:  [376.31171602]
    ---------------------------------------------------------------------------------------------------
    Trial number: 14500
    The total training accuracy:  [376.31098605]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15000
    The total training accuracy:  [376.31008674]
    ---------------------------------------------------------------------------------------------------
    Trial number: 15500
    The total training accuracy:  [376.30897939]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16000
    The total training accuracy:  [376.30761721]
    ---------------------------------------------------------------------------------------------------
    Trial number: 16500
    The total training accuracy:  [376.30594414]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17000
    The total training accuracy:  [376.3038938]
    ---------------------------------------------------------------------------------------------------
    Trial number: 17500
    The total training accuracy:  [376.3013887]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18000
    The total training accuracy:  [376.2983402]
    ---------------------------------------------------------------------------------------------------
    Trial number: 18500
    The total training accuracy:  [376.29464943]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19000
    The total training accuracy:  [376.29020986]
    ---------------------------------------------------------------------------------------------------
    Trial number: 19500
    The total training accuracy:  [376.28491169]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20000
    The total training accuracy:  [376.27864771]
    ---------------------------------------------------------------------------------------------------
    Trial number: 20500
    The total training accuracy:  [376.27131896]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21000
    The total training accuracy:  [376.26283601]
    ---------------------------------------------------------------------------------------------------
    Trial number: 21500
    The total training accuracy:  [376.25311056]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22000
    The total training accuracy:  [376.24203439]
    ---------------------------------------------------------------------------------------------------
    Trial number: 22500
    The total training accuracy:  [376.22945141]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23000
    The total training accuracy:  [376.2172902]
    ---------------------------------------------------------------------------------------------------
    Trial number: 23500
    The total training accuracy:  [376.20513033]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24000
    The total training accuracy:  [376.19311376]
    ---------------------------------------------------------------------------------------------------
    Trial number: 24500
    The total training accuracy:  [376.18111812]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25000
    The total training accuracy:  [376.16909987]
    ---------------------------------------------------------------------------------------------------
    Trial number: 25500
    The total training accuracy:  [376.1790345]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26000
    The total training accuracy:  [376.14522337]
    ---------------------------------------------------------------------------------------------------
    Trial number: 26500
    The total training accuracy:  [376.13362634]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27000
    The total training accuracy:  [376.12200661]
    ---------------------------------------------------------------------------------------------------
    Trial number: 27500
    The total training accuracy:  [376.11034372]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28000
    The total training accuracy:  [376.09872607]
    ---------------------------------------------------------------------------------------------------
    Trial number: 28500
    The total training accuracy:  [376.08713768]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29000
    The total training accuracy:  [376.0755948]
    ---------------------------------------------------------------------------------------------------
    Trial number: 29500
    The total training accuracy:  [376.06411502]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30000
    The total training accuracy:  [376.06042445]
    ---------------------------------------------------------------------------------------------------
    Trial number: 30500
    The total training accuracy:  [377.4791601]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31000
    The total training accuracy:  [376.03010564]
    ---------------------------------------------------------------------------------------------------
    Trial number: 31500
    The total training accuracy:  [376.01895837]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32000
    The total training accuracy:  [376.00782946]
    ---------------------------------------------------------------------------------------------------
    Trial number: 32500
    The total training accuracy:  [375.99680705]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33000
    The total training accuracy:  [375.98582319]
    ---------------------------------------------------------------------------------------------------
    Trial number: 33500
    The total training accuracy:  [375.97487267]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34000
    The total training accuracy:  [375.96396405]
    ---------------------------------------------------------------------------------------------------
    Trial number: 34500
    The total training accuracy:  [375.95318612]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35000
    The total training accuracy:  [375.94242954]
    ---------------------------------------------------------------------------------------------------
    Trial number: 35500
    The total training accuracy:  [375.93171842]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36000
    The total training accuracy:  [375.92105176]
    ---------------------------------------------------------------------------------------------------
    Trial number: 36500
    The total training accuracy:  [375.91045324]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37000
    The total training accuracy:  [375.89990252]
    ---------------------------------------------------------------------------------------------------
    Trial number: 37500
    The total training accuracy:  [375.88940963]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38000
    The total training accuracy:  [375.87895909]
    ---------------------------------------------------------------------------------------------------
    Trial number: 38500
    The total training accuracy:  [375.86856514]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39000
    The total training accuracy:  [375.85821352]
    ---------------------------------------------------------------------------------------------------
    Trial number: 39500
    The total training accuracy:  [375.84792721]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40000
    The total training accuracy:  [375.83770689]
    ---------------------------------------------------------------------------------------------------
    Trial number: 40500
    The total training accuracy:  [375.82759142]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41000
    The total training accuracy:  [375.81749177]
    ---------------------------------------------------------------------------------------------------
    Trial number: 41500
    The total training accuracy:  [375.80743845]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42000
    The total training accuracy:  [375.7974224]
    ---------------------------------------------------------------------------------------------------
    Trial number: 42500
    The total training accuracy:  [375.78744358]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43000
    The total training accuracy:  [375.77750537]
    ---------------------------------------------------------------------------------------------------
    Trial number: 43500
    The total training accuracy:  [375.76761245]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44000
    The total training accuracy:  [375.75775787]
    ---------------------------------------------------------------------------------------------------
    Trial number: 44500
    The total training accuracy:  [375.74794881]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45000
    The total training accuracy:  [375.73817472]
    ---------------------------------------------------------------------------------------------------
    Trial number: 45500
    The total training accuracy:  [375.7284582]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46000
    The total training accuracy:  [375.71880367]
    ---------------------------------------------------------------------------------------------------
    Trial number: 46500
    The total training accuracy:  [375.70917842]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47000
    The total training accuracy:  [375.69960227]
    ---------------------------------------------------------------------------------------------------
    Trial number: 47500
    The total training accuracy:  [375.69005181]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48000
    The total training accuracy:  [375.68055671]
    ---------------------------------------------------------------------------------------------------
    Trial number: 48500
    The total training accuracy:  [375.67108786]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49000
    The total training accuracy:  [375.66164632]
    ---------------------------------------------------------------------------------------------------
    Trial number: 49500
    The total training accuracy:  [375.65223089]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50000
    The total training accuracy:  [375.64285226]
    ---------------------------------------------------------------------------------------------------
    Trial number: 50500
    The total training accuracy:  [375.6335533]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51000
    The total training accuracy:  [375.62427643]
    ---------------------------------------------------------------------------------------------------
    Trial number: 51500
    The total training accuracy:  [375.61502585]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52000
    The total training accuracy:  [375.60583499]
    ---------------------------------------------------------------------------------------------------
    Trial number: 52500
    The total training accuracy:  [375.59665789]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53000
    The total training accuracy:  [375.58749857]
    ---------------------------------------------------------------------------------------------------
    Trial number: 53500
    The total training accuracy:  [375.57837631]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54000
    The total training accuracy:  [375.56928678]
    ---------------------------------------------------------------------------------------------------
    Trial number: 54500
    The total training accuracy:  [375.56023736]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55000
    The total training accuracy:  [375.55122375]
    ---------------------------------------------------------------------------------------------------
    Trial number: 55500
    The total training accuracy:  [375.54222856]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56000
    The total training accuracy:  [375.53325978]
    ---------------------------------------------------------------------------------------------------
    Trial number: 56500
    The total training accuracy:  [375.52576531]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57000
    The total training accuracy:  [375.57828461]
    ---------------------------------------------------------------------------------------------------
    Trial number: 57500
    The total training accuracy:  [375.51469871]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58000
    The total training accuracy:  [375.49817103]
    ---------------------------------------------------------------------------------------------------
    Trial number: 58500
    The total training accuracy:  [375.489034]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59000
    The total training accuracy:  [375.48022727]
    ---------------------------------------------------------------------------------------------------
    Trial number: 59500
    The total training accuracy:  [375.47149446]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60000
    The total training accuracy:  [375.46278848]
    ---------------------------------------------------------------------------------------------------
    Trial number: 60500
    The total training accuracy:  [375.45410841]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61000
    The total training accuracy:  [375.44545455]
    ---------------------------------------------------------------------------------------------------
    Trial number: 61500
    The total training accuracy:  [375.43682341]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62000
    The total training accuracy:  [375.428211]
    ---------------------------------------------------------------------------------------------------
    Trial number: 62500
    The total training accuracy:  [375.41963446]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63000
    The total training accuracy:  [375.41107428]
    ---------------------------------------------------------------------------------------------------
    Trial number: 63500
    The total training accuracy:  [375.40252968]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64000
    The total training accuracy:  [375.39401387]
    ---------------------------------------------------------------------------------------------------
    Trial number: 64500
    The total training accuracy:  [375.38551084]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65000
    The total training accuracy:  [375.37705339]
    ---------------------------------------------------------------------------------------------------
    Trial number: 65500
    The total training accuracy:  [375.36860688]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66000
    The total training accuracy:  [375.36023554]
    ---------------------------------------------------------------------------------------------------
    Trial number: 66500
    The total training accuracy:  [375.35187196]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67000
    The total training accuracy:  [375.34351156]
    ---------------------------------------------------------------------------------------------------
    Trial number: 67500
    The total training accuracy:  [375.33521486]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68000
    The total training accuracy:  [375.32694971]
    ---------------------------------------------------------------------------------------------------
    Trial number: 68500
    The total training accuracy:  [375.31868845]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69000
    The total training accuracy:  [375.3104362]
    ---------------------------------------------------------------------------------------------------
    Trial number: 69500
    The total training accuracy:  [375.30221514]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70000
    The total training accuracy:  [375.29406753]
    ---------------------------------------------------------------------------------------------------
    Trial number: 70500
    The total training accuracy:  [375.28593107]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71000
    The total training accuracy:  [375.27781975]
    ---------------------------------------------------------------------------------------------------
    Trial number: 71500
    The total training accuracy:  [375.26970329]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72000
    The total training accuracy:  [375.26163673]
    ---------------------------------------------------------------------------------------------------
    Trial number: 72500
    The total training accuracy:  [375.2536039]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73000
    The total training accuracy:  [375.24555915]
    ---------------------------------------------------------------------------------------------------
    Trial number: 73500
    The total training accuracy:  [375.23752417]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74000
    The total training accuracy:  [375.22949988]
    ---------------------------------------------------------------------------------------------------
    Trial number: 74500
    The total training accuracy:  [375.22149294]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75000
    The total training accuracy:  [375.21350108]
    ---------------------------------------------------------------------------------------------------
    Trial number: 75500
    The total training accuracy:  [375.20552988]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76000
    The total training accuracy:  [375.19757171]
    ---------------------------------------------------------------------------------------------------
    Trial number: 76500
    The total training accuracy:  [375.1896405]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77000
    The total training accuracy:  [375.18172922]
    ---------------------------------------------------------------------------------------------------
    Trial number: 77500
    The total training accuracy:  [375.17382997]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78000
    The total training accuracy:  [375.16594339]
    ---------------------------------------------------------------------------------------------------
    Trial number: 78500
    The total training accuracy:  [375.15811569]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79000
    The total training accuracy:  [375.15032141]
    ---------------------------------------------------------------------------------------------------
    Trial number: 79500
    The total training accuracy:  [375.14253005]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80000
    The total training accuracy:  [375.13474812]
    ---------------------------------------------------------------------------------------------------
    Trial number: 80500
    The total training accuracy:  [375.1269777]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81000
    The total training accuracy:  [375.11922204]
    ---------------------------------------------------------------------------------------------------
    Trial number: 81500
    The total training accuracy:  [375.11147816]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82000
    The total training accuracy:  [375.10374568]
    ---------------------------------------------------------------------------------------------------
    Trial number: 82500
    The total training accuracy:  [375.09602329]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83000
    The total training accuracy:  [375.08835133]
    ---------------------------------------------------------------------------------------------------
    Trial number: 83500
    The total training accuracy:  [375.08068846]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84000
    The total training accuracy:  [375.07304124]
    ---------------------------------------------------------------------------------------------------
    Trial number: 84500
    The total training accuracy:  [375.06540362]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85000
    The total training accuracy:  [375.05781632]
    ---------------------------------------------------------------------------------------------------
    Trial number: 85500
    The total training accuracy:  [375.05023915]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86000
    The total training accuracy:  [375.04267017]
    ---------------------------------------------------------------------------------------------------
    Trial number: 86500
    The total training accuracy:  [375.03511345]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87000
    The total training accuracy:  [375.02757202]
    ---------------------------------------------------------------------------------------------------
    Trial number: 87500
    The total training accuracy:  [375.02004183]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88000
    The total training accuracy:  [375.01252528]
    ---------------------------------------------------------------------------------------------------
    Trial number: 88500
    The total training accuracy:  [375.00502051]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89000
    The total training accuracy:  [374.99752715]
    ---------------------------------------------------------------------------------------------------
    Trial number: 89500
    The total training accuracy:  [374.99004209]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90000
    The total training accuracy:  [374.98258892]
    ---------------------------------------------------------------------------------------------------
    Trial number: 90500
    The total training accuracy:  [374.9751473]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91000
    The total training accuracy:  [374.96771323]
    ---------------------------------------------------------------------------------------------------
    Trial number: 91500
    The total training accuracy:  [374.96029348]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92000
    The total training accuracy:  [374.95291214]
    ---------------------------------------------------------------------------------------------------
    Trial number: 92500
    The total training accuracy:  [374.9455476]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93000
    The total training accuracy:  [374.9381818]
    ---------------------------------------------------------------------------------------------------
    Trial number: 93500
    The total training accuracy:  [374.930832]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94000
    The total training accuracy:  [374.92350388]
    ---------------------------------------------------------------------------------------------------
    Trial number: 94500
    The total training accuracy:  [374.91617512]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95000
    The total training accuracy:  [374.90884744]
    ---------------------------------------------------------------------------------------------------
    Trial number: 95500
    The total training accuracy:  [374.90152275]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96000
    The total training accuracy:  [374.89423482]
    ---------------------------------------------------------------------------------------------------
    Trial number: 96500
    The total training accuracy:  [374.88698785]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97000
    The total training accuracy:  [374.87974387]
    ---------------------------------------------------------------------------------------------------
    Trial number: 97500
    The total training accuracy:  [374.87251539]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98000
    The total training accuracy:  [374.86528423]
    ---------------------------------------------------------------------------------------------------
    Trial number: 98500
    The total training accuracy:  [374.85811981]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99000
    The total training accuracy:  [374.85095653]
    ---------------------------------------------------------------------------------------------------
    Trial number: 99500
    The total training accuracy:  [374.84378082]
    ---------------------------------------------------------------------------------------------------
    Trial number: 100000
    The total training accuracy:  [374.83660592]
    ---------------------------------------------------------------------------------------------------
    


```python
plot_curve = []
v = 4
dv = (13-4)/300

vestas_2mw_power_out = lambda v: (vestas_2mw_std_model.predict(polynomial_transform(np.array(v,ndmin=2),5,[]))[0,0])
for i in range(300):
    plot_curve.append(vestas_2mw_power_out(v))
    v = v + dv

# plt.plot(power_out_data)
plt.plot(plot_curve)
```




    [<matplotlib.lines.Line2D at 0x223577ca9a0>]




    
![png](output_55_1.png)
    


##### A 1 layer NN model with 'tanh' activation to fit the Power Curve:


```python
power_out_data = vestas_2mw_data.loc[:,1.225]

X_train = power_out_data.index.values.reshape(1, power_out_data.size)

Y_train = power_out_data.values.reshape(1, power_out_data.size)

tanh_2mw_model = NN(2,[1, 6, 1], ["tanh", "None", ], losses.mse_loss, losses.mse_loss_deriv,AdaM_opt=True)
```


```python
# Hyperparameter tuning:
tanh_2mw_model.reglr_type = "L1"
tanh_2mw_model.lmbd = 0

# tanh_2mw_model.dropout_probs[1, 0] = 1

tanh_2mw_model.frwrd_trials = 10000
tanh_2mw_model.back_trials = 10000

# Training:
total_trials = 100001
batch_size = X_train.shape[1] // 1

for trial in range(total_trials):

    # Hyperparameter tuning:
    tanh_2mw_model.alpha = 0.0001

    # Stochastic GD 'mask' :
    mask = np.random.choice(X_train.shape[1], batch_size, replace=False)

    train_loss = (NN.training_cycle(tanh_2mw_model, X=X_train[:,mask], Y=Y_train[:,mask]))[0]
    if((trial)%1000==0):
        print("Trial number:",trial)
        print("The total training loss: ",train_loss)
        print("---------------------------------------------------------------------------------------------------")
```


```python
plot_curve = []
v = 0
dv = (30-0)/300

tanh_2mw_power_out = lambda v: np.where(v<4,0,np.where(v<=13.5,(tanh_2mw_model.predict(v)[0,0]),np.where(v<=25,2000,0)))/1000
for i in range(300):
    plot_curve.append(tanh_2mw_power_out(v))
    v = v + dv

# plt.plot(power_out_data)
plt.plot(plot_curve)
```




    [<matplotlib.lines.Line2D at 0x14c17152910>]




    
![png](output_59_1.png)
    


##### Testing the model on a 80 WT Farm having Horns Rev 1 layout with given Wind Speed data:


```python
# Description of the Wind Farm layout and Turbine specification:

# WT specifications
rotor_D, hub_h, Ct, alpha = 90, 100, 0.3, 0.08

# Wind Farm layout:
theta = 0
seperation  = 7 * rotor_D
rows = 8
columns = 10

# Wind velocities:
start_date = '2006-01-01 00:00'
end_date = '2007-01-01 00:00'
yr_vels = windpower_data.loc[start_date:end_date]["Wind Speed (m/s)"]

# Initializing WT's according to layout and specifications.
turbines = np.empty(shape=(rows,columns),dtype=np.dtype("object"))
for i in range(rows):
    for j in range(columns):

        # Assigning the position of WT on plane (Grid configuration)
        x_co = seperation*j
        y_co = seperation*i
        
        # # Another configuration (Alternate Shifted Grid)
        # x_co = seperation*j 
        # y_co = seperation*i + seperation*(j%2)/2
        
        turbines[i,j] = WindTurbine(rotor_D,hub_h,Ct,alpha,x_co,y_co,0)

turbines = turbines.T.flatten() # Change the layout of the array from 2D to 1D.

power_table = {}
avg_power_yr = 0

power_table_no_wake = {}
avg_power_yr_no_wake = 0

# timer = 0
for V in yr_vels:

    # timer += 1
    # print(timer/yr_vels.size*100,"%")

    if(power_table_no_wake.get(V,None)==None):
        power_table_no_wake[V] = 0
        for wt in turbines:
            power_table_no_wake[V] += tanh_2mw_power_out(V)

    avg_power_yr_no_wake += power_table_no_wake[V]

    if(power_table.get(V,None)!=None):
        avg_power_yr += power_table[V]
        continue

    for wt in turbines:
        wt.V = V
    
    for wt in turbines:
        calc_WT_vel_superposition(wt,turbines)
    
    power = 0
    for wt in turbines:
        power += tanh_2mw_power_out(wt.V)

    power_table[V] = power
    avg_power_yr += power_table[V]

avg_power_yr = avg_power_yr / yr_vels.size
avg_power_yr_no_wake = avg_power_yr_no_wake / yr_vels.size

print("Average Wind Velocity:",np.mean(yr_vels),"m/s")
print("Average Farm Output Power (With Wake Effect)",avg_power_yr,"MW")
print("Average Farm Output Power (Without Wake Effect)",avg_power_yr_no_wake,"MW")
```

    Average Wind Velocity: 7.792312785388127 m/s
    Average Farm Output Power (With Wake Effect) 64.32824124705796 MW
    Average Farm Output Power (Without Wake Effect) 71.65905021010194 MW
    

### Study of Fatigue Damage in Wind Turbines

##### Defining Pre-processing techniques : (Hysteresis filtering, Peak-Valley Filtering, Binning)


```python
def hysteresis_filtering(load_spectra, band_width = 1):
    if(len(load_spectra)==0):
        return load_spectra
    
    final_spectra = []
    level = load_spectra[0]

    for i in range(len(load_spectra)):
        if(abs(load_spectra[i] - level) >= band_width):
            level = load_spectra[i]
        final_spectra.append(level)
    
    return final_spectra

def binning_data(load_spectra, bin_count = 29):
    bin_base = min(load_spectra)
    bin_width = (max(load_spectra) + 1 - bin_base) / bin_count # '+1' to avoid singular peak.
    
    if(bin_width==0):
        return load_spectra.copy()
    
    final_spectra = []
    for i in range(len(load_spectra)):
        final_spectra.append(int((load_spectra[i]-bin_base) // bin_width))

    return (final_spectra, bin_width, bin_base)

# Doubt: Should inflection points other than peaks and valleys not be removed?
def peak_valley_filtering(load_spectra):
    if(len(load_spectra)<=1):
        return load_spectra.copy()
    
    final_spectra = [load_spectra[0], load_spectra[1]]
    for i in range(2,len(load_spectra)):
        flag = final_spectra[-1]-final_spectra[-2]
        if(flag>0):
            if(load_spectra[i]<final_spectra[-1]):
                final_spectra.append(load_spectra[i])
            else:
                final_spectra[-1] = load_spectra[i]
        elif(flag<0):
            if(load_spectra[i]>final_spectra[-1]):
                final_spectra.append(load_spectra[i])
            else:
                final_spectra[-1] = load_spectra[i]
        else:
            final_spectra[-1] = load_spectra[i]

    return final_spectra

```

##### Defining the Rainflow Counting Algorithm and 'S-N' Curve


```python
def SN_curve(p_range, m = -5.76, k = 3.24, Su = 390):
    # 'S' being the stress-range of the cycle in mega pascals(MPa).(Not normalized)
    # log(p_range) = log(c) + (1/m)*log(N) ;
    # N = (p_range/c)**m ;
    
    if(p_range==0):
        return np.inf
    
    return (p_range/(k*Su))**m



class linked_list(object):
    def __init__(self,value,next=None,prev=None) -> None:
        self.val = value
        self.nxt = next
        self.prv = prev
    
    def insert(self,value):
        if(self==None):
            return linked_list(value)
        
        x = linked_list(value,self.nxt,self)
        
        if(self.nxt!=None):
            self.nxt.prv = x
        self.nxt = x
        
        return x
    
    def delete(self):
        if(self==None):
            return None
        
        if(self.nxt!=None):
            self.nxt.prv = self.prv
        if(self.prv!=None):
            self.prv.nxt = self.nxt

        r = (self.prv) if (self.prv!=None and self.nxt==None) else (self.nxt)
        del self

        return r

def rainflow_counting(data,bins):
    rainflow_matrix = [[0 for i in range(bins)] for h in range(bins)]

    # damage_history[i][0] denotes stress-range, while damage_history[i][1] denotes stress-mean 
    # of a full cycle encountered in the load history.
    damage_history = [(0,0)]

    start = end = linked_list(data[0])
    d_ptr = 1

    while(d_ptr<len(data)):
        if(start==end or start.nxt==end or start.nxt.nxt==end):
            end = end.insert(data[d_ptr])
            d_ptr += 1

        while not (start==end or start.nxt==end or start.nxt.nxt==end):
            o1 = start.val
            i1 = start.nxt.val
            i2 = start.nxt.nxt.val
            o2 = start.nxt.nxt.nxt.val
            
            flag = True
            if(o1>o2):
                o1,o2 = o2,o1
            if(i1>i2):
                i1,i2 = i2,i1
                flag = False

            if(o1<=i1 and o2>=i2):
                if(flag):
                    rainflow_matrix[i1][i2] += 1
                else:
                    rainflow_matrix[i2][i1] += 1
                
                damage_history.append(((i2-i1), (i2+i1)//2))

                start.nxt.nxt.delete()
                start.nxt.delete()
            
                if(start.prv!=None):
                    start = start.prv
                if(start.prv!=None):
                    start = start.prv
            else:
                start = start.nxt
                damage_history.append((0,0))

    residue = []
    while(end!=None):
        residue.append(end.val)
        end = end.delete()
    
    for i in range(len(residue)//2):
        residue[i], residue[len(residue)-1-i] = residue[len(residue)-1-i], residue[i]
    
    return rainflow_matrix, damage_history, residue
```

##### Verification of the model and some results



```python
sig, mu = 89, 0
data = np.random.randn(15000)*sig + mu

bins = 101
hyst_noise_band = 0

processed_data, bin_width, base_value = binning_data(peak_valley_filtering(hysteresis_filtering(data,hyst_noise_band)),bins)
processed_data = peak_valley_filtering(processed_data)

print("Bin Width =",bin_width,"; Base Value =",base_value)

plt.figure(figsize=(17,6))
plt.plot(processed_data)
plt.show()
```

    Bin Width = 7.0237179939266445 ; Base Value = -346.489231948206
    


    
![png](output_68_1.png)
    



```python
rf_matrix, history, residue = rainflow_counting(processed_data,bins)

plt.plot(residue)
plt.show()
```


    
![png](output_69_0.png)
    



```python
fatigue_dmg = [0]
for dat in history:
    fatigue_dmg.append(fatigue_dmg[-1] + (1 / SN_curve(dat[0]*bin_width)) )

plt.plot(fatigue_dmg)
plt.show()
```


    
![png](output_70_0.png)
    



```python
sns.heatmap(rf_matrix,linewidth=0)
plt.show()
```


    
![png](output_71_0.png)
    



```python

```
