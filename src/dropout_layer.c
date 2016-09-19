#include "dropout_layer.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
}

void forward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
    if (!state.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) state.input[i] = 0;
        else state.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
    if(!state.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) state.delta[i] = 0;
        else state.delta[i] *= l.scale;
    }
}

