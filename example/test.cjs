const { promisify } = require('util')
const { readFile } = require('fs')
const path=require('path')
const loadFile = promisify(readFile)
const SVM =  require('../dist/svm.js').default
/**
 * svm train and predict
 */
start();

async function loadData(file) {
  let data = [], labels = []
  return await loadFile(path.resolve(__dirname, file)).then(res => {
    let temp = []
    res.toString().split(/ +|\r\n+/).forEach((v, i) => {
      if (i % 3 == 2) {
        labels.push(v)
        data.push(temp)
        temp = []
      }
      else temp.push(v)
    })
    return { data, labels }
  })
}
async function start() {
  const { data, labels } = await loadData('train_data')
  const { data:testdata, labels:testlabels } = await loadData('test_data')
  const svm=new SVM()
  svm.train(data,labels)
  console.log(`${'-'.repeat(20)}\n predict start!`);
  var result=svm.predict(testdata)
  console.log(result);
}


