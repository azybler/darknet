#include "layer.h"
#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
        return;
    }
    if(l.indexes)        free(l.indexes);
    if(l.rand)           free(l.rand);
    if(l.cost)           free(l.cost);
    if(l.filters)        free(l.filters);
    if(l.filter_updates) free(l.filter_updates);
    if(l.biases)         free(l.biases);
    if(l.bias_updates)   free(l.bias_updates);
    if(l.weights)        free(l.weights);
    if(l.weight_updates) free(l.weight_updates);
    if(l.col_image)      free(l.col_image);
    if(l.input_layers)   free(l.input_layers);
    if(l.input_sizes)    free(l.input_sizes);
    if(l.delta)          free(l.delta);
    if(l.output)         free(l.output);
    if(l.squared)        free(l.squared);
    if(l.norms)          free(l.norms);
}
