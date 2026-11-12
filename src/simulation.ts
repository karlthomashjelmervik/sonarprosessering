export type WindowType = 'Rectangular' | 'Hamming'
export type ElementView = 'Magnitude' | 'Phase' | 'Real' | 'Imaginary'

export interface Parameters {
  width: number
  elementCountX: number
  frequency: number
  bandwidth: number
  pulseDuration: number
  prr: number
  fanAngle: number
  sourceLevel: number
  reflectorEnabled: boolean
  reflectorStrengthDb: number
  window: WindowType
  trainingCells: number
  guardCells: number
  threshold: number
}

export interface Scatterer {
  x: number
  y: number
}

export interface SimulationBase {
  nx: number
  ny: number
  totalElements: number
  simElementCount: number
  wavelength: number
  sampleRate: number
  rangeResolution: number
  beamwidthH: number
  beamSpacing: number
  beamCountLimited: boolean
  beamwidthV: number
  directivity: number
  directivityIndex: number
  maxUnambiguousRange: number
  estimatedSnr: number
  rangeAxis: number[]
  angleAxis: number[]
  targetOutline: Scatterer[]
  scatterers: Scatterer[]
  reflector: Scatterer | undefined
  elementX: number[]
  elementZ: number[]
  elementDataReal: Float32Array
  elementDataImag: Float32Array
  elementDataRows: number
  elementDataCols: number
  beamformed: Float32Array
  matched: Float32Array
  rows: number
  cols: number
}

export interface CfarResults {
  cfar: Float32Array
  truthMask: Uint8Array
  signalPower: Float32Array
}

export interface DetectionResults {
  detections: Int8Array
  detectionCount: number
  falseAlarmCount: number
  falseAlarmRate: number
  targetSnrDb: number | undefined
}

export type Simulation = SimulationBase & CfarResults & DetectionResults

export const SOUND_SPEED = 1500
export const MAX_ARRAY_ELEMENTS = 192
const SCENE_MAX_RANGE = 110
const MAX_SIM_ELEMENTS = 96
const MAX_BEAMFORMING_CELLS = 250_000
const STAR_CENTER_RANGE = 80
const TARGET_SIZE_SCALE = 5
const TARGET_SCATTER_SPACING = 0.35
const TARGET_STRENGTH_DB = -35

function thorpAbsorptionDbPerKm(frequencyHz: number): number {
  const f = frequencyHz / 1000
  const f2 = f * f
  return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003
}

function starScatterers(): { outline: Scatterer[]; scatterers: Scatterer[]; totalScattererCount: number } {
  const vertices: Scatterer[] = []
  for (let i = 0; i < 10; i += 1) {
    const radius = (i % 2 === 0 ? 2.5 : 1.05) * TARGET_SIZE_SCALE
    const angle = -Math.PI / 2 + i * Math.PI / 5
    vertices.push({ x: Math.cos(angle) * radius, y: STAR_CENTER_RANGE + Math.sin(angle) * radius })
  }

  const outline: Scatterer[] = []
  const points: Scatterer[] = []
  for (let i = 0; i < vertices.length; i += 1) {
    const from = vertices[i]
    const to = vertices[(i + 1) % vertices.length]
    outline.push(from)
    const edgeLength = Math.hypot(to.x - from.x, to.y - from.y)
    const steps = Math.ceil(edgeLength / TARGET_SCATTER_SPACING)
    for (let step = 0; step < steps; step += 1) {
      const t = step / steps
      points.push({ x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t })
    }
  }

  const visibleScatterers = points.filter((point) => {
    const pointSquared = point.x * point.x + point.y * point.y
    for (let i = 0; i < vertices.length; i += 1) {
      const from = vertices[i]
      const to = vertices[(i + 1) % vertices.length]
      const edgeX = to.x - from.x
      const edgeY = to.y - from.y
      const denominator = point.x * edgeY - point.y * edgeX
      const epsilon = 1e-9

      if (Math.abs(denominator) <= epsilon) {
        const collinearity = from.x * point.y - from.y * point.x
        if (Math.abs(collinearity) <= epsilon) {
          const fromProjection = (from.x * point.x + from.y * point.y) / pointSquared
          const toProjection = (to.x * point.x + to.y * point.y) / pointSquared
          if (Math.min(fromProjection, toProjection) >= 0
            && Math.min(fromProjection, toProjection) < 1 - epsilon
            && Math.max(fromProjection, toProjection) >= 0) return false
        }
        continue
      }

      const intersectionRange = (from.x * edgeY - from.y * edgeX) / denominator
      const edgePosition = (from.x * point.y - from.y * point.x) / denominator
      if (intersectionRange >= 0 && intersectionRange < 1 - epsilon
        && edgePosition >= -epsilon && edgePosition <= 1 + epsilon) return false
    }
    return true
  })

  return { outline, scatterers: visibleScatterers, totalScattererCount: points.length }
}

function positionsInAperture(count: number, apertureExtent: number, maxCount: number): number[] {
  const simulatedCount = Math.min(count, maxCount)
  if (simulatedCount <= 1) return [0]
  return Array.from({ length: simulatedCount }, (_, index) => -apertureExtent / 2 + apertureExtent * index / (simulatedCount - 1))
}

function representativeGridCounts(nx: number, ny: number): { nx: number; ny: number } {
  if (nx * ny <= MAX_SIM_ELEMENTS) return { nx, ny }
  if (nx <= MAX_SIM_ELEMENTS) return { nx, ny: Math.max(1, Math.floor(MAX_SIM_ELEMENTS / nx)) }
  if (ny <= MAX_SIM_ELEMENTS) return { nx: Math.max(1, Math.floor(MAX_SIM_ELEMENTS / ny)), ny }
  return { nx: 12, ny: 8 }
}

function hammingWeight(position: number, extent: number, enabled: boolean): number {
  if (!enabled || extent === 0) return 1
  const normalized = position / extent + 0.5
  return 0.54 - 0.46 * Math.cos(2 * Math.PI * normalized)
}

function deterministicNoise(seed: number): () => number {
  let state = seed >>> 0
  let spare: number | undefined
  return () => {
    if (spare !== undefined) {
      const value = spare
      spare = undefined
      return value
    }
    let u = 0
    let v = 0
    while (u === 0) u = ((state = (1664525 * state + 1013904223) >>> 0) + 1) / 4294967297
    while (v === 0) v = ((state = (1664525 * state + 1013904223) >>> 0) + 1) / 4294967297
    const magnitude = Math.sqrt(-2 * Math.log(u))
    const angle = 2 * Math.PI * v
    spare = magnitude * Math.sin(angle)
    return magnitude * Math.cos(angle)
  }
}

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined
  values.sort((a, b) => a - b)
  const middle = Math.floor(values.length / 2)
  return values.length % 2 === 0
    ? (values[middle - 1] + values[middle]) / 2
    : values[middle]
}

export function simulate(parameters: Parameters): SimulationBase {
  const wavelength = SOUND_SPEED / parameters.frequency
  const nx = Math.min(MAX_ARRAY_ELEMENTS, Math.max(1, Math.floor(parameters.elementCountX)))
  const ny = 1
  const totalElements = nx * ny
  const beamwidthHRad = 0.886 * wavelength / parameters.width
  const beamwidthVRad = Math.PI
  const beamwidthH = beamwidthHRad * 180 / Math.PI
  const beamwidthV = 180
  const directivity = 4 * Math.PI / Math.max(beamwidthHRad * beamwidthVRad, 1e-12)
  const directivityIndex = 10 * Math.log10(directivity)
  const maxUnambiguousRange = SOUND_SPEED / (2 * parameters.prr)

  const fullElementXPositions = positionsInAperture(nx, parameters.width, nx)
  const fullElementZPositions = positionsInAperture(ny, 0, ny)
  const elementX: number[] = []
  const elementZ: number[] = []
  for (const z of fullElementZPositions) {
    for (const x of fullElementXPositions) {
      elementX.push(x)
      elementZ.push(z)
    }
  }
  const simulatedGrid = representativeGridCounts(nx, ny)
  const simulatedElementXPositions = positionsInAperture(nx, parameters.width, simulatedGrid.nx)
  const simulatedElementZPositions = positionsInAperture(ny, 0, simulatedGrid.ny)
  const simulatedElementX: number[] = []
  const simulatedElementZ: number[] = []
  for (const z of simulatedElementZPositions) {
    for (const x of simulatedElementXPositions) {
      simulatedElementX.push(x)
      simulatedElementZ.push(z)
    }
  }
  const elementCount = simulatedElementX.length
  const simulatedWeights = simulatedElementX.map((x, index) => (
    hammingWeight(x, parameters.width, parameters.window === 'Hamming')
    * hammingWeight(simulatedElementZ[index], 0, parameters.window === 'Hamming')
  ))
  const weightSum = simulatedWeights.reduce((sum, weight) => sum + weight, 0)
  const fullHorizontalWeights = fullElementXPositions.map((x) => (
    hammingWeight(x, parameters.width, parameters.window === 'Hamming')
  ))
  const fullHorizontalWeightSum = Math.max(fullHorizontalWeights.reduce((sum, weight) => sum + weight, 0), 1e-12)
  const k = 2 * Math.PI / wavelength
  const horizontalArrayResponse = (directionSine: number) => {
    let real = 0
    let imag = 0
    for (let element = 0; element < nx; element += 1) {
      const phase = k * fullElementXPositions[element] * directionSine
      real += fullHorizontalWeights[element] * Math.cos(phase)
      imag += fullHorizontalWeights[element] * Math.sin(phase)
    }
    return { real: real / fullHorizontalWeightSum, imag: imag / fullHorizontalWeightSum }
  }

  const bandwidth = Math.min(parameters.bandwidth, parameters.frequency * 0.5)
  const sampleRate = bandwidth * 1.1
  const rangeStep = SOUND_SPEED / (2 * sampleRate)
  const rangeBinCount = Math.floor(SCENE_MAX_RANGE / rangeStep) + 1
  const rangeAxis = Array.from({ length: rangeBinCount }, (_, i) => i * rangeStep)
  const desiredBeamCount = Math.ceil(parameters.fanAngle / (beamwidthH / 2)) + 1
  const maxBeamCount = Math.max(2, Math.floor(MAX_BEAMFORMING_CELLS / rangeBinCount))
  const beamCount = Math.min(desiredBeamCount, maxBeamCount)
  const beamCountLimited = beamCount < desiredBeamCount
  const beamSpacing = parameters.fanAngle / (beamCount - 1)
  const angleAxis = Array.from(
    { length: beamCount },
    (_, i) => -parameters.fanAngle / 2 + i * beamSpacing,
  )
  const rows = angleAxis.length
  const cols = rangeBinCount
  const { outline: targetOutline, scatterers, totalScattererCount } = starScatterers()
  const reflector = parameters.reflectorEnabled
    ? { x: 0, y: 55 }
    : undefined
  const echoSources = reflector ? [...scatterers, reflector] : scatterers
  const elementDataReal = new Float32Array(elementCount * rangeBinCount)
  const elementDataImag = new Float32Array(elementCount * rangeBinCount)
  const cleanReal = new Float32Array(elementDataReal.length)
  const cleanImag = new Float32Array(elementDataImag.length)
  const timeStep = 1 / sampleRate
  const pulseDurationSeconds = parameters.pulseDuration / 1000
  const pulseSamples = Math.max(1, Math.min(rangeBinCount - 1, Math.round(pulseDurationSeconds / timeStep)))
  const pulseReal = new Float32Array(pulseSamples)
  const pulseImag = new Float32Array(pulseSamples)
  for (let i = 0; i < pulseSamples; i += 1) {
    const t = i * timeStep
    const phase = Math.PI * bandwidth / pulseDurationSeconds * (t * t - pulseDurationSeconds * t)
    pulseReal[i] = Math.cos(phase)
    pulseImag[i] = Math.sin(phase)
  }
  const absorption = thorpAbsorptionDbPerKm(parameters.frequency)
  const cleanBeamformed = new Float32Array(rows * cols * 2)

  for (const point of echoSources) {
    const range = Math.hypot(point.x, point.y)
    const angle = Math.atan2(point.x, point.y)
    const observedRange = ((range % maxUnambiguousRange) + maxUnambiguousRange) % maxUnambiguousRange
    const startSample = Math.round(observedRange / rangeStep)
    const spreadingLoss = 40 * Math.log10(Math.max(range, 1))
    const absorptionLoss = 2 * (range / 1000) * absorption
    const relativeStrengthDb = point === reflector ? parameters.reflectorStrengthDb : 0
    const amplitude = 10 ** ((parameters.sourceLevel + TARGET_STRENGTH_DB + relativeStrengthDb - spreadingLoss - absorptionLoss - 150) / 20)
      / Math.sqrt(totalScattererCount)
    // Sector-wide transmission illuminates the full target; receive beamforming supplies angular selectivity.
    const txGain = 1

    for (let element = 0; element < elementCount; element += 1) {
      const indexBase = element * rangeBinCount
      const receivePhase = -k * simulatedElementX[element] * Math.sin(angle)
      const phase = -4 * Math.PI * range / wavelength + receivePhase
      const coefficientReal = amplitude * txGain * Math.cos(phase)
      const coefficientImag = amplitude * txGain * Math.sin(phase)
      const available = Math.min(pulseSamples, rangeBinCount - startSample)
      for (let sample = 0; sample < available; sample += 1) {
        const index = indexBase + startSample + sample
        cleanReal[index] += coefficientReal * pulseReal[sample] - coefficientImag * pulseImag[sample]
        cleanImag[index] += coefficientReal * pulseImag[sample] + coefficientImag * pulseReal[sample]
      }
    }

    const available = Math.min(pulseSamples, rangeBinCount - startSample)
    for (let beam = 0; beam < rows; beam += 1) {
      const steering = angleAxis[beam] * Math.PI / 180
      const receiveResponse = horizontalArrayResponse(Math.sin(steering) - Math.sin(angle))
      const phase = -4 * Math.PI * range / wavelength
      const coefficientReal = amplitude * txGain * (
        receiveResponse.real * Math.cos(phase) - receiveResponse.imag * Math.sin(phase)
      )
      const coefficientImag = amplitude * txGain * (
        receiveResponse.real * Math.sin(phase) + receiveResponse.imag * Math.cos(phase)
      )
      const rowOffset = beam * cols * 2
      for (let sample = 0; sample < available; sample += 1) {
        const index = rowOffset + (startSample + sample) * 2
        cleanBeamformed[index] += coefficientReal * pulseReal[sample] - coefficientImag * pulseImag[sample]
        cleanBeamformed[index + 1] += coefficientReal * pulseImag[sample] + coefficientImag * pulseReal[sample]
      }
    }
  }

  const noiseDb = 65 + 20 * Math.log10(parameters.frequency / 100_000) + 10 * Math.log10(Math.max(bandwidth, 1) / 8000)
  const noiseSigma = 10 ** ((noiseDb - 150) / 20)
  const random = deterministicNoise(45821)
  let signalPeak = 0
  for (let i = 0; i < elementDataReal.length; i += 1) {
    const realNoise = random() * noiseSigma / Math.sqrt(2)
    const imagNoise = random() * noiseSigma / Math.sqrt(2)
    elementDataReal[i] = cleanReal[i] + realNoise
    elementDataImag[i] = cleanImag[i] + imagNoise
    signalPeak = Math.max(signalPeak, Math.hypot(cleanReal[i], cleanImag[i]))
  }
  const estimatedSnr = 20 * Math.log10(Math.max(signalPeak, 1e-15) / Math.max(noiseSigma, 1e-15))

  const beamformed = new Float32Array(rows * cols * 2)
  for (let beam = 0; beam < rows; beam += 1) {
    const steering = angleAxis[beam] * Math.PI / 180
    const rowOffset = beam * cols * 2
    for (let element = 0; element < elementCount; element += 1) {
      const phase = k * simulatedElementX[element] * Math.sin(steering)
      const steerReal = Math.cos(phase)
      const steerImag = Math.sin(phase)
      const weight = simulatedWeights[element] / Math.max(weightSum, 1e-12)
      const inputBase = element * rangeBinCount
      for (let sample = 0; sample < cols; sample += 1) {
        const inputReal = elementDataReal[inputBase + sample] - cleanReal[inputBase + sample]
        const inputImag = elementDataImag[inputBase + sample] - cleanImag[inputBase + sample]
        const index = rowOffset + sample * 2
        beamformed[index] += weight * (inputReal * steerReal - inputImag * steerImag)
        beamformed[index + 1] += weight * (inputReal * steerImag + inputImag * steerReal)
      }
    }
  }
  for (let index = 0; index < beamformed.length; index += 1) {
    beamformed[index] += cleanBeamformed[index]
  }

  const matched = new Float32Array(beamformed.length)
  let pulseEnergy = 0
  for (let i = 0; i < pulseSamples; i += 1) pulseEnergy += pulseReal[i] ** 2 + pulseImag[i] ** 2
  const normalization = Math.max(pulseEnergy, 1e-12)
  for (let beam = 0; beam < rows; beam += 1) {
    const rowOffset = beam * cols * 2
    for (let sample = 0; sample < cols; sample += 1) {
      let sumReal = 0
      let sumImag = 0
      const available = Math.min(pulseSamples, cols - sample)
      for (let pulse = 0; pulse < available; pulse += 1) {
        const signalIndex = rowOffset + (sample + pulse) * 2
        sumReal += beamformed[signalIndex] * pulseReal[pulse] + beamformed[signalIndex + 1] * pulseImag[pulse]
        sumImag += beamformed[signalIndex + 1] * pulseReal[pulse] - beamformed[signalIndex] * pulseImag[pulse]
      }
      const outputIndex = rowOffset + sample * 2
      matched[outputIndex] = sumReal / normalization
      matched[outputIndex + 1] = sumImag / normalization
    }
  }

  return {
    nx, ny, totalElements, simElementCount: elementCount,
    wavelength, sampleRate, rangeResolution: SOUND_SPEED / (2 * bandwidth), beamwidthH, beamSpacing, beamCountLimited, beamwidthV, directivity, directivityIndex,
    maxUnambiguousRange, estimatedSnr,
    rangeAxis, angleAxis, targetOutline, scatterers, reflector, elementX, elementZ, elementDataReal, elementDataImag,
    elementDataRows: elementCount, elementDataCols: rangeBinCount,
    beamformed, matched, rows, cols,
  }
}

export function calculateCfar(
  simulation: SimulationBase,
  parameters: Pick<Parameters, 'trainingCells' | 'guardCells'>,
): CfarResults {
  const { matched, rows, cols, scatterers, reflector, maxUnambiguousRange, rangeAxis, angleAxis, beamwidthH } = simulation
  const cfar = new Float32Array(rows * cols)
  const truthMask = new Uint8Array(rows * cols)
  const signalPower = new Float32Array(rows * cols)
  const halfTraining = Math.max(1, Math.floor(parameters.trainingCells / 2))
  const rangeStep = rangeAxis[1] - rangeAxis[0]
  const truthRangeBins: number[] = []
  const truthAngles: number[] = []
  for (const point of reflector ? [...scatterers, reflector] : scatterers) {
    const range = Math.hypot(point.x, point.y)
    truthRangeBins.push(Math.round((((range % maxUnambiguousRange) + maxUnambiguousRange) % maxUnambiguousRange) / rangeStep))
    truthAngles.push(Math.atan2(point.x, point.y) * 180 / Math.PI)
  }
  for (let beam = 0; beam < rows; beam += 1) {
    const angle = angleAxis[beam]
    const rowOffset = beam * cols * 2
    const cellOffset = beam * cols
    for (let sample = 0; sample < cols; sample += 1) {
      const index = rowOffset + sample * 2
      const cell = cellOffset + sample
      const real = matched[index]
      const imag = matched[index + 1]
      const power = real * real + imag * imag
      signalPower[cell] = power
      let noisePower = 0
      let count = 0
      for (let side = -1; side <= 1; side += 2) {
        for (let offset = parameters.guardCells + 1; offset <= parameters.guardCells + halfTraining; offset += 1) {
          const trainingIndex = sample + side * offset
          if (trainingIndex >= 0 && trainingIndex < cols) {
            const trainingComplex = rowOffset + trainingIndex * 2
            noisePower += matched[trainingComplex] ** 2 + matched[trainingComplex + 1] ** 2
            count += 1
          }
        }
      }
      const normalized = power / Math.max(noisePower / Math.max(count, 1), 1e-18)
      cfar[cell] = normalized
      let isTruth = false
      for (let target = 0; target < truthRangeBins.length; target += 1) {
        if (Math.abs(sample - truthRangeBins[target]) <= 2
          && Math.abs(angle - truthAngles[target]) <= Math.max(beamwidthH / 2, 0.5)) {
          isTruth = true
          break
        }
      }
      truthMask[cell] = isTruth ? 1 : 0
    }
  }

  return { cfar, truthMask, signalPower }
}

export function calculateDetections(
  cfarResults: CfarResults,
  threshold: number,
): DetectionResults {
  const { cfar, truthMask, signalPower } = cfarResults
  const detections = new Int8Array(cfar.length)
  const backgroundPowers: number[] = []
  let maximumTargetPower = 0
  let detectionCount = 0
  let falseAlarmCount = 0
  for (let cell = 0; cell < cfar.length; cell += 1) {
    if (truthMask[cell]) maximumTargetPower = Math.max(maximumTargetPower, signalPower[cell])
    else backgroundPowers.push(signalPower[cell])
    if (cfar[cell] > threshold) {
      detections[cell] = truthMask[cell] ? 1 : -1
      detectionCount += 1
      if (!truthMask[cell]) falseAlarmCount += 1
    }
  }
  const falseAlarmRate = falseAlarmCount / cfar.length
  const medianBackgroundPower = median(backgroundPowers)
  const targetSnrDb = medianBackgroundPower !== undefined
    && maximumTargetPower > 0
    && medianBackgroundPower > 0
    ? 10 * Math.log10(maximumTargetPower / medianBackgroundPower)
    : undefined
  return { detections, detectionCount, falseAlarmCount, falseAlarmRate, targetSnrDb }
}

export function plotMatrix(
  simulation: SimulationBase,
  view: ElementView,
): { z: number[][]; x: number[]; y: number[]; title: string; colorbarTitle: string } {
  const { elementDataRows: rows, elementDataCols: cols, elementDataReal, elementDataImag } = simulation
  const stride = 4
  const sampleIndexes = Array.from({ length: Math.ceil(cols / stride) }, (_, index) => index * stride)
  const data = Array.from({ length: rows }, (_, row) => sampleIndexes.map((sample) => {
    const index = row * cols + sample
    const real = elementDataReal[index]
    const imag = elementDataImag[index]
    if (view === 'Real') return 20 * Math.log10(Math.max(Math.abs(real), 1e-12))
    if (view === 'Imaginary') return 20 * Math.log10(Math.max(Math.abs(imag), 1e-12))
    if (view === 'Phase') return Math.atan2(imag, real)
    return 20 * Math.log10(Math.max(Math.hypot(real, imag), 1e-12))
  }))
  const labels = {
    Magnitude: ['Element magnitude', 'Level (dB)'],
    Phase: ['Element phase', 'Phase (rad)'],
    Real: ['Real component magnitude', 'Level (dB)'],
    Imaginary: ['Imaginary component magnitude', 'Level (dB)'],
  } as const
  return {
    z: data,
    x: sampleIndexes.map((sample) => simulation.rangeAxis[sample]),
    y: Array.from({ length: rows }, (_, row) => row),
    title: labels[view][0],
    colorbarTitle: labels[view][1],
  }
}

export function resultMatrix(simulation: Simulation, mode: 'beam' | 'matched' | 'cfar' | 'detections'): number[][] {
  return Array.from({ length: simulation.rows }, (_, row) => {
    const rowBase = row * simulation.cols
    return Array.from({ length: simulation.cols }, (_, col) => {
      const cell = rowBase + col
      if (mode === 'cfar') return 10 * Math.log10(Math.max(simulation.cfar[cell], 1e-12))
      if (mode === 'detections') return simulation.detections[cell]
      const complexIndex = cell * 2
      const values = mode === 'beam' ? simulation.beamformed : simulation.matched
      const magnitude = Math.hypot(values[complexIndex], values[complexIndex + 1])
      return 20 * Math.log10(Math.max(magnitude, 1e-12))
    })
  })
}

export function cartesianMatrix(
  polarMatrix: number[][],
  rangeAxis: number[],
  angleAxis: number[],
  nearest = false,
): { z: number[][]; axis: number[] } {
  const gridSize = 201
  const maxRange = rangeAxis[rangeAxis.length - 1]
  const axis = Array.from({ length: gridSize }, (_, index) => -maxRange + 2 * maxRange * index / (gridSize - 1))
  const rangeStep = rangeAxis[1] - rangeAxis[0]
  const angleStep = angleAxis[1] - angleAxis[0]
  const z = axis.map((y) => axis.map((x) => {
    const range = Math.hypot(x, y)
    if (range > maxRange) return Number.NaN

    const angle = Math.atan2(x, y) * 180 / Math.PI
    const rangePosition = (range - rangeAxis[0]) / rangeStep
    const anglePosition = (angle - angleAxis[0]) / angleStep
    const rangeIndex = nearest ? Math.round(rangePosition) : Math.floor(rangePosition)
    const angleIndex = nearest ? Math.round(anglePosition) : Math.floor(anglePosition)
    if (nearest) return polarMatrix[angleIndex]?.[rangeIndex] ?? Number.NaN

    const nextRangeIndex = Math.min(rangeIndex + 1, rangeAxis.length - 1)
    const nextAngleIndex = Math.min(angleIndex + 1, angleAxis.length - 1)
    const rangeFraction = rangePosition - rangeIndex
    const angleFraction = anglePosition - angleIndex
    const topLeft = polarMatrix[angleIndex]?.[rangeIndex] ?? Number.NaN
    const topRight = polarMatrix[angleIndex]?.[nextRangeIndex] ?? Number.NaN
    const bottomLeft = polarMatrix[nextAngleIndex]?.[rangeIndex] ?? Number.NaN
    const bottomRight = polarMatrix[nextAngleIndex]?.[nextRangeIndex] ?? Number.NaN
    return (topLeft * (1 - rangeFraction) + topRight * rangeFraction) * (1 - angleFraction)
      + (bottomLeft * (1 - rangeFraction) + bottomRight * rangeFraction) * angleFraction
  }))

  return { z, axis }
}
