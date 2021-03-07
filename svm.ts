export default class SVM {
    public data!: mat
    public labels!: number[]

    public C: number
    public tol: number
    private maxIter: number
    private kernelType: string
    private numChange:number

    public alpha: number[]=[]
    public b: number=0
    public row: number = 0
    public col: number = 0
  constructor(option?:Options) {
    const {C = 1, toler = 1e-4, maxIter = 1e4, kernelType = 'linear',numChange=10}=option||{}
    this.C = C
    this.tol = toler
    this.maxIter = maxIter
    this.kernelType = kernelType
    this.numChange=numChange
  }
  /**
   * @param data  [1,2,3,4] 
   * @param label [1]
   */
  public train(data: matrix<any>, label: any[]) {
    if (data.length !== label.length) {
      throw new Error('Features and labels should have the same length');
    }
    this.data = data.map(_=>_.map(v=>parseFloat(v)))
    this.labels = label.map(v=>parseFloat(v))
    this.row = data.length
    this.col = data[0].length
    this.alpha = new Array(this.row).fill(0)
    this.smo()
    return this
  }
  private predictOne(dat: number[]) {
    var f = this.b
    for (let i = 0; i < this.col; i++) {
      var s = 0.0;
      for (var j = 0; j < this.row; j++) {
        s += this.alpha[j] * this.labels[j] * this.data[j][i];
      }
      f += dat[i]  * s;
    }
    return f > 0 ? 1 : -1
  }

  public predict(data: matrix<any>) {
    data = data.map(_=>_.map(v=>parseFloat(v)))
    var result = new Array(this.col)
    for (let i = 0; i < this.col; i++)
      result[i] = this.predictOne(data[i])
    return result
  }

  private kernel(i: number, j: number): number {
    let s = 0
    // linear
    if (this.kernelType == 'linear') {
      for (let n = 0; n < this.col; n++) {
        s += this.data[i][n] * this.data[j][n]
      }
    }
    return s;
  }
  /**
   * @description 计算g(xi)与yi之差
   */
  private calcFx(k: number): number {
    var f = this.b;
    for (var i = 0; i < this.row; i++) {
      f += this.alpha[i]  * this.labels[i] * this.kernel(k, i)
    }
    return f
  }

  private update(i: number): number {
    var L = 0, H = 0, eta
    var Ei = this.calcFx(i) - this.labels[i]
    /**
     * 不符合kkt条件 进行优化 yi*Ei=yi*gx-yi^2=yi*gx-1
     * alpha=0    yi*gx-1 >= 0
     * 0<alpha<C  yi*gx-1 == 0
     * alpha=C    yi*gx-1 <= 0
     */
    if (this.labels[i] * Ei < -this.tol && this.alpha[i] < this.C || 0 < this.alpha[i] && this.labels[i] * Ei > this.tol) {
      var j = i
      while (j == i) {
        j = Math.floor(Math.random() * this.row);
      }
      var Ej = this.calcFx(j) - this.labels[j]
      var ai = this.alpha[i];
      var aj = this.alpha[j];
      if (this.labels[i] === this.labels[j]) {
        L = Math.max(0, ai + aj - this.C);
        H = Math.min(this.C, ai + aj);
      } else {
        L = Math.max(0, aj - ai);
        H = Math.min(this.C, this.C + aj + ai);
      }
      if (Math.abs(L - H) < 1e-4) {
        return 0;
      }
      // calcuate eta=2K12-K11-K22
      eta = 2 * this.kernel(i, j) - this.kernel(i, i) - this.kernel(j, j);
      if (eta >= 0) return 0
      // new alpha2 = {H (>H) | newaj  |L (<L)}
      var newaj = aj - this.labels[j] * (Ei - Ej) / eta;
      if (newaj > H) newaj = H
      else if (newaj < L) newaj = L
      //目标函数需要有足够的下降
      if (Math.abs(newaj - aj) < this.tol) return 0
      var newai = ai + this.labels[i] * this.labels[j] * (aj - newaj)
      this.alpha[i] = newai
      this.alpha[j] = newaj
      var newb1 = -Ei - this.labels[i] * this.kernel(i, i) * (newai - ai) - this.labels[j] * this.kernel(2, 1) * (newaj - aj) + this.b
      var newb2 = -Ej - this.labels[i] * this.kernel(i, j) * (newai - ai) - this.labels[j] * this.kernel(2, 2) * (newaj - aj) + this.b
      if (0 < newai && newai < this.C)
        this.b = newb1
      else if (0 < newaj && newaj < this.C)
        this.b = newb2
      else
        this.b = (newb1 + newb2) *0.5
      return 1
    }else{
      return 0
    }
  }
  private smo() {
    var iter = 0
    var alphaChange = 0
    var num=0
    while (iter < this.maxIter && num < this.numChange ) {
        for (let i = 0; i < this.row; i++)
          alphaChange += this.update(i)
        iter++
      if (alphaChange==0) num++
      else num=0
    }
  }
}
