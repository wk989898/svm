type matrix<T> = T[][]
type mat = matrix<number>
interface Array<T> {
  transponse(): matrix<unknown> | Array<unknown>
}
type kernelType ='linear' |'rbf'
