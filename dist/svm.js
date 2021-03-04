"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class SVM {
    constructor(C = 1, toler = 1e-4, maxIter = 10 ^ 4, kernelType = 'linear') {
        this.len = 0;
        this.alpha = Array(this.len).fill(0);
        this.C = C;
        this.tol = toler;
        this.b = 0;
        this.maxIter = maxIter;
        this.kernelType = kernelType;
    }
    /**
     * @param data  [1,2,3,4]
     * @param label [1]
     */
    train(data, label) {
        this.data = data;
        this.labels = label;
        this.len = data.length;
        this.smo();
    }
    predict() {
    }
    kernel(i, j) {
        let s = 0;
        // linear
        if (this.kernelType == 'linear') {
            for (let n = 0; n < this.len; n++) {
                s += this.data[i][n] * this.data[j][n];
            }
        }
        return s;
    }
    /**
     * @description 计算g(xi)与yi之差
     */
    calcE(k) {
        var f = this.b;
        for (var i = 0; i < this.len; i++) {
            f += this.alpha[i] * this.labels[i] * this.kernel(k, i);
        }
        return f - this.labels[k];
    }
    update(i) {
        var L = 0, H = 0, eta;
        var Ei = this.calcE(i);
        /**
         * 不符合kkt条件 进行优化 yi*Ei=yi*gx-yi^2=yi*gx-1
         * alpha=0    yi*gx-1 >= 0
         * 0<alpha<C  yi*gx-1 == 0
         * alpha=C    yi*gx-1 <= 0
         */
        if (this.labels[i] * Ei < -this.tol && this.alpha[i] < this.C || 0 < this.alpha[i] && this.labels[i] * Ei > this.tol) {
            var j = i;
            while (j == i) {
                j = Math.floor(Math.random() * this.len);
            }
            var Ej = this.calcE(j);
            var ai = this.alpha[i];
            var aj = this.alpha[j];
            if (this.labels[i] === this.labels[j]) {
                L = Math.max(0, ai + aj - this.C);
                H = Math.min(this.C, ai + aj);
            }
            else {
                L = Math.max(0, aj - ai);
                H = Math.min(this.C, this.C + aj + ai);
            }
            if (Math.abs(L - H) < 1e-4) {
                return 0;
            }
            // calcuate eta=2K12-K11-K22
            eta = 2 * this.kernel(i, j) - this.kernel(i, i) - this.kernel(j, j);
            if (eta >= 0)
                return 0;
            // new alpha2 = {H (>H) | newaj  |L (<L)}
            var newaj = aj - this.labels[j] * (Ei - Ej) / eta;
            if (newaj > H)
                newaj = H;
            else if (newaj < L)
                newaj = L;
            //目标函数需要有足够的下降
            if (Math.abs(newaj - aj) < this.tol)
                return 0;
            var newai = ai + this.labels[i] * this.labels[j] * (aj - newaj);
            var newb1 = -Ei - this.labels[i] * this.kernel(i, i) * (newai - ai) - this.labels[j] * this.kernel(2, 1) * (newaj - aj) + this.b;
            var newb2 = -Ej - this.labels[i] * this.kernel(i, j) * (newai - ai) - this.labels[j] * this.kernel(2, 2) * (newaj - aj) + this.b;
            if (0 < newb1 && newb1 < this.C)
                this.b = newb1;
            else if (0 < newb2 && newb2 < this.C)
                this.b = newb2;
            else
                this.b = (newb1 + newb2) / 2;
            this.alpha[i] = newai;
            this.alpha[j] = newaj;
            return 1;
        }
        return 0;
    }
    smo() {
        var iter = 0;
        var alphaChange = 0, entry = true;
        while (iter < this.maxIter && (alphaChange > 0 || entry)) {
            if (entry) {
                for (let i = 0; i < this.len; i++)
                    alphaChange += this.update(i);
                console.log(`alpha changed! now is ${alphaChange} times`);
            }
            else {
                for (let i = 0; i < this.len; i++) {
                    if (this.alpha[i] < 0 || this.alpha[i] > this.C)
                        alphaChange += this.update(i);
                    console.log(`alpha changed! now is ${alphaChange} times`);
                }
            }
            entry = false;
        }
    }
}
exports.default = SVM;
