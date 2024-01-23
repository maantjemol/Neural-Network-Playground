<script lang="ts">
  export let label: string = ''
  export let value: string | number= ''
</script>

<div class="my-2">
  <label>
    <p class="mb-2">{label}</p>
    <input {...$$restProps} class="px-2 py-1 border-white-2 text-white rounded-mb bg-black border-violet-600 border-2 rounded-md" bind:value />
  </label>
</div>