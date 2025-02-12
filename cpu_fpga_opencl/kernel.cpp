__kernel void ndvi_kernel(__global const float* nir, 
                        __global const float* red, 
                        __global float* ndvi)
{   
    // get index of the work item
    
    int index = get_global_id(0);
    
    float nir_value = nir[index];
    float red_value = red[index];
    ndvi[index] = ((nir_value + red_value) <= 0) ? 0 : (nir_value - red_value) / (nir_value + red_value);
    
}
