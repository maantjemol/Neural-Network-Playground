<script lang="ts">
  export let value: number = 0
</script>

<div class="w-full rounded-sm h-2.5 bg-gray-700">
  <div class="bg-violet-400 h-2.5 rounded-sm duration-200 ease-out" style={`width: ${value}%`}></div>
</div>