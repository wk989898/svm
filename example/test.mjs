import {promisify} from 'util'
import {readFile} from 'fs'
const loadFile = promisify(readFile);

loadFile('./train_data').then(res=>{
  console.log(res);
}).catch(console.error)
const SVM=require('../dist/svm.js')
// var svm=new SVM()