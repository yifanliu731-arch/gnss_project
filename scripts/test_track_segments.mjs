import assert from 'node:assert/strict'
import { splitTrack } from '../src/trackSegments.ts'

const points = [0,1,2,3,4,5].map(i=>({x:i,y:0,gap_before:i===2,transfer:i===3}))
const result = splitTrack(points)
function edges(data) {
  return data.slice(1).flatMap((b,i)=>data[i] && b ? [[data[i][0],b[0]]] : [])
}
assert.deepEqual(edges(result.track),[[0,1],[2,3],[4,5]])
assert.deepEqual(edges(result.gap),[[1,2]])
assert.deepEqual(edges(result.transfer),[[3,4]])
assert.equal(Object.values(result).flatMap(edges).length,points.length-1)
console.log('Discontinuous tracks preserve every edge without bridging solid sections.')
