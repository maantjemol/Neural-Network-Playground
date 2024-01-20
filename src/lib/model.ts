type Actfs = "tanh" | "relu" | "sig";

export class Value {
  public data: number;
  public grad: number;
  public label: string;
  private _prev: Value[];
  private _op: string;
  private _backward: () => void;

  constructor(data: number, _children: Value[] = [], _op = "", label = "") {
    this.data = data;
    this._prev = _children;
    this.grad = 0.0;
    this._op = _op;
    this._backward = () => {};
    this.label = label;
  }

  toString() {
    return `Value(data=${this.data})`;
  }

  add(value: Value | number) {
    const other = typeof value === "number" ? new Value(value) : value;

    const out = new Value(this.data + other.data, [this, other], "+");

    const _backward = () => {
      this.grad += 1 * out.grad;
      other.grad += 1 * out.grad;
    };

    out._backward = _backward;

    return out;
  }

  multiply(value: Value | number) {
    const other = typeof value === "number" ? new Value(value) : value;
    const out = new Value(this.data * other.data, [this, other], "*");

    const _backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };

    out._backward = _backward;

    return out;
  }

  pow(value: number) {
    const out = new Value(this.data ** value, [this], `**${value}`);

    const _backward = () => {
      this.grad += value * this.data ** (value - 1) * out.grad;
    };

    out._backward = _backward;

    return out;
  }

  neg() {
    return this.multiply(-1);
  }

  subtract(value: Value | number) {
    const other = typeof value === "number" ? new Value(value) : value;
    return this.add(other.neg());
  }

  divide(value: Value) {
    const other = typeof value === "number" ? new Value(value) : value;
    return this.multiply(other.pow(-1));
  }

  tanh() {
    const x = this.data;
    const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);

    const out = new Value(t, [this], "tanh");

    const _backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  sig() {
    const x = this.data;
    const r = 1 / (1 + Math.exp(-x));

    const out = new Value(r, [this], "sig");

    const _backward = () => {
      this.grad += r * (1 - r) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  relu() {
    const x = this.data;
    const r = x > 0 ? x : 0;

    const out = new Value(r, [this], "relu");

    const _backward = () => {
      this.grad = (x > 0 ? 1 : 0) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        v._prev.forEach((child) => {
          buildTopo(child);
        });
        topo.push(v);
      }
    };

    this.grad = 1;
    buildTopo(this);

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
}

export class Neuron {
  private w: Value[];
  private b: Value;
  private actf: Actfs;

  constructor(nin: number, actf: Actfs = "tanh") {
    this.w = Array(nin)
      .fill(0)
      .map(() => new Value(Math.random() * 2 - 1));
    this.b = new Value(Math.random() * 2 - 1);
    this.actf = actf;
  }

  forward(x: Value[]): Value {
    const act = this.w.reduce(
      (sum, wi, i) => sum.add(wi.multiply(x[i])),
      this.b
    );

    let out: Value;

    switch (this.actf) {
      case "relu":
        out = act.relu();
        break;
      case "sig":
        out = act.sig();
        break;
      default:
        out = act.tanh();
    }

    return out;
  }

  parameters() {
    return [...this.w, this.b];
  }
}

class Layer {
  private neurons: Neuron[];

  constructor(nin: number, nout: number, actf: Actfs = "tanh") {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, actf));
  }

  forward(x: Value[]) {
    const outs = this.neurons.map((neuron) => {
      return neuron.forward(x);
    });
    return outs;
  }

  parameters() {
    const params: Value[] = [];
    this.neurons.forEach((neuron) => {
      const ps = neuron.parameters();
      params.push(...ps);
    });
    return params;
  }
}

export class MLP {
  private layers: Layer[];

  constructor(nin: number, nouts: number[], actfs: Actfs[]) {
    const sz = [nin, ...nouts];
    this.layers = sz
      .slice(0, -1)
      .map((nin, i) => new Layer(nin, sz[i + 1], actfs[i]));
  }

  forward(x: Value[]) {
    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];
      x = layer.forward(x);
    }

    return x;
  }

  parameters() {
    const params: Value[] = [];
    this.layers.forEach((layer) => {
      const ps = layer.parameters();
      params.push(...ps);
    });
    return params;
  }
}

export const SGD = (n: MLP, stepsize: number = 0.01) => {
  n.parameters().forEach((p) => (p.data += -stepsize * p.grad));
};

export function convertToValues(xs: number[][]) {
  return xs.map((subArray) => subArray.map((num) => new Value(num)));
}
