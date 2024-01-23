<script lang="ts">
  let buttonProps = {
	  class:[$$restProps.class]
	}
</script>

<button on:click
  {...buttonProps}
  class="bg-violet-600 rounded-md px-2 py-1" 
>
  <slot/>
</button>