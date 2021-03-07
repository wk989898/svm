type matrix<T> = T[][]
type mat = matrix<number>
type kernelType ='linear' |'rbf'
interface Options{
   C: number
   b: number
   tol: number
   alpha: number[]
   row: number 
   col: number 
   toler:number
   maxIter: number
   kernelType: string
   numChange:number
}
