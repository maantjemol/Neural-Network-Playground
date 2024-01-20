<script lang="ts">
  import '@carbon/charts-svelte/styles.css';
  import { LineChart, ScaleTypes } from '@carbon/charts-svelte'
  import { MLP, Value, SGD, convertToValues } from "$lib/model";
  import { onMount } from 'svelte';
  import {xs, ys} from "$lib/fruitdata";

  let lossdata:{}[] = [];
  let loading = false;
  let stepnum = 0;

  let stepnumbers = 200;
  let learningrate = 0.05;
  let prediction: Value[] = [];


  let index = 0;

  let n = new MLP(4, [10, 10, 4], ["tanh", "relu", "tanh"]);

  const xsValues = convertToValues(xs);

  $: prediction = n.forward(xsValues[index])

  const runModel = (n: MLP) => {
    loading = true;
  
    // const xs = [
    //   [0, 0],
    //   [1, 0],
    //   [0, 1],
    //   [1, 1],
    // ];

    // const ys: number[][] = [[1.0], [-1.0], [-1.0], [1.0]];

    

    for (let k = 0; k < stepnumbers; k++) {
      const ypred = xsValues.map((x) => n.forward(x));

      let loss = new Value(0);
      ypred.forEach((pred, i) => {
        pred.forEach((val, j) => {
          loss = loss.add(val.subtract(ys[i][j]).pow(2));
        });
      });

      n.parameters().forEach((p) => {
        p.grad = 0;
      });

      loss.backward();

      SGD(n, learningrate);
      console.log(loss.data);
      

      stepnum = k;
      lossdata.push({
        group: 'loss',
        value: loss.data,
        step: k
      })
    }
    loading = false;
  }

  let options = {
		title: 'Loss',
    axes: {
      bottom: {
        title: 'step',
        mapsTo: 'step',
        scaleType: "linear",
      },
      left: {
        mapsTo: 'value',
        title: 'Loss',
        scaleType: "linear",
      },
    },
    points: {
      radius: 0,  
      enabled: false
    },
    data: {
      loading: loading,
    }
	}

  runModel(n);
  
</script>

<h1>Model:</h1>
<button on:click={() => {
  lossdata = [];
  n = new MLP(4, [10, 10, 4], ["tanh", "tanh", "tanh"]);
  runModel(n);
}}>Rerun</button>
<p>Amount of steps</p>
<input type="number" bind:value={stepnumbers} min="1" max="1000" step="1" />
<p>Learning rate</p>
<input type="number" bind:value={learningrate} min="0.01" max="1" step="0.01" />
<LineChart
  data={lossdata}
  bind:options={options} 
/>

<h1>Test model:</h1>
<button on:click={() => index += 1}>Get sample</button>
<p>Result: {ys[index]}</p>
<p>Prediction</p>
<p>Apple: {Math.round(prediction[0].data * 100)}%</p>
<p>Mandarin: {Math.round(prediction[1].data * 100)}%</p>
<p>Orange: {Math.round(prediction[2].data * 100)}%</p>
<p>Lemon: {Math.round(prediction[3].data * 100)}%</p>
