# multi-signal-preprocess

This is column-wise apply filter and utility process module

:warning: Warning Butterworth may not work fluently

## How to use

#### import preprocess file

<code>
import preprocess as pp
</code>

#### choose which filter you want to apply

<code>
  filter = pp.HampelFiltering() 
</code>

#### set condition of filter
this .__condition__ part is optional

.__condition__ will only return information of filter condition, and update filter condition. 

as it is not necessary, it return nothing but update the condition value
  
if you did not assign a condition, it will set the condition of filter as default

<code>
filter_condition = filter.__condition__(window_size=200, n_sigmas=2, percentile=0.75, torch=True, device=device)
</code>

#### put data in it

<code>
  
        processed_data = filter.filtering(data)
</code>

this line will make way more simpler

### data type
this function apply all its function along axis in every columns.

simply said, If your data is consists of number_of_sensors and number_of_data,

shape of data should be [numbe_of_sensors, number_of_data]
