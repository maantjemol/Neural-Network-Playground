<script lang="ts">
  import '@carbon/charts-svelte/styles.css';
  import { LineChart, ScaleTypes } from '@carbon/charts-svelte'
  import { MLP, Value, SGD, convertToValues } from "$lib/model";
  import { onMount } from 'svelte';
  import {xs, ys} from "$lib/fruitdata";
  import Button from '../components/Button.svelte';
  import Input from '../components/Input.svelte';
  import Meter from '../components/Meter.svelte';

  let lossdata:{}[] = [];
  let loading = true;
  let stepnum = 0;

  let stepnumbers = 500;
  let learningrate = 0.005;
  let prediction: Value[] = [];

  const fruitData = ["Apple 🍎", "Mandarin 🍊", "Orange 🍊", "Lemon 🍋"]

  let index = 0;

  let n = new MLP(4, [10, 10, 4], ["tanh", "relu", "tanh"]);

  const xsValues = convertToValues(xs);
  const xtValues = convertToValues(xs);

  $: prediction = n.forward(xtValues[index])

  const runModel = (n: MLP) => {
    loading = true;
    for (let k = 0; k < stepnumbers; k++) {
      stepnum = k;
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
    theme: "g100",
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
	}

  runModel(n);
  
</script>
<!-- Settings -->
{#if loading}
  <div class="bg-black opacity-50 w-full h-full fixed top-0 right-0 z-50"/>
  <div class="flex w-screen h-screen fixed top-0 right-0 items-center justify-center z-50">
    <p class="opacity-100">Training model...</p>
  </div>
{/if}

<section class="w-4/5 mx-auto mb-24 mt-12">
  <h1 class="text-2xl font-bold">Settings:</h1>
  <hr class="w-full border-gray-400 border-b-2 mt-2 mb-4">
  <Input type="number" bind:value={stepnumbers} min="1" max="1000" step="1" label="Steps/epochs (amount of times the model sees the data)"/>
  <Input type="number" bind:value={learningrate} min="0.01" max="1" step="0.01" label="Learning rate (how much the model adjusts to changes):"/>

  <hr class="w-full border-gray-500 border-b-2 mt-4 mb-2">

  <Button
  on:click={() => {
    lossdata = [];
    n = new MLP(4, [10, 10, 4], ["tanh", "tanh", "tanh"]);
    runModel(n);
  }}>Rerun</Button>
<!-- Charts -->
<h1 class="text-2xl font-bold mt-12">Charts:</h1>
<hr class="w-full border-gray-400 border-b-2 mt-1 mb-2">
<LineChart
  data={lossdata}
  bind:options={options} 
/>

<h1 class="text-2xl font-bold mt-12">Results:</h1>
<hr class="w-full border-gray-400 border-b-2 mt-1 mb-2">
<p>Expected result: {fruitData[ys[index].indexOf(1)]} ({index})</p>
<p class="mt-4">Prediction: {fruitData[prediction.findIndex((t) => t.data === Math.max(...prediction.map((p) => p.data)))]}</p>
<p class="mt-1 mb-1">Apple: {Math.max(Math.round(prediction[0].data * 100),0)}%</p>
<Meter value={Math.max(Math.round(prediction[0].data * 100),0)} />
<p class="mt-1 mb-1">Mandarin: {Math.max(Math.round(prediction[1].data * 100),0)}%</p>
<Meter value={Math.max(Math.round(prediction[1].data * 100),0)} />
<p class="mt-1 mb-1">Orange: {Math.max(Math.round(prediction[2].data * 100),0)}%</p>
<Meter value={Math.max(Math.round(prediction[2].data * 100),0)} />
<p class="mt-1 mb-1">Lemon: {Math.max(Math.round(prediction[3].data * 100),0)}%</p> 
<Meter value={Math.max(Math.round(prediction[3].data * 100),0)} />
<hr class="w-full border-gray-500 border-b-2 mt-4 mb-2">

<Button
  on:click={() => {
    index = Math.floor(Math.random() * ys.length);
  }}>Randomize</Button>
</section>
