import { useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import type { Config, Data, Layout } from 'plotly.js'
import {
  calculateCfar,
  calculateDetections,
  cartesianMatrix,
  plotMatrix,
  resultMatrix,
  MAX_ARRAY_ELEMENTS,
  SOUND_SPEED,
  simulate,
  type ElementView,
  type Parameters,
  type Simulation,
  type WindowType,
} from './simulation'

const initialParameters: Parameters = {
  width: 0.8,
  elementCountX: 112,
  frequency: 100_000,
  bandwidth: 8_000,
  pulseDuration: 8,
  prr: 5,
  fanAngle: 30,
  sourceLevel: 205,
  reflectorEnabled: false,
  reflectorStrengthDb: 30,
  window: 'Hamming',
  trainingCells: 16,
  guardCells: 64,
  threshold: 12,
}

type ResultMode = 'beam' | 'matched' | 'cfar' | 'detections'
type ColorBounds = { min: string; max: string }

const emptyColorBounds: ColorBounds = { min: '', max: '' }
const MIN_APERTURE_WIDTH = 0.1
const MAX_APERTURE_WIDTH = 5
const MAX_CFAR_WINDOW_METERS = 10

function halfWavelengthAtBandEdge(frequency: number, bandwidth: number): number {
  return SOUND_SPEED / (2 * (frequency + bandwidth / 2))
}

function elementCountForAperture(width: number, frequency: number, bandwidth: number): number {
  return Math.min(MAX_ARRAY_ELEMENTS, Math.ceil(width / halfWavelengthAtBandEdge(frequency, bandwidth)) + 1)
}

function linkedArrayForAperture(width: number, frequency: number, bandwidth: number) {
  const elementSpacing = halfWavelengthAtBandEdge(frequency, bandwidth)
  const maxWidth = (MAX_ARRAY_ELEMENTS - 1) * elementSpacing
  const apertureWidth = Math.min(width, Math.floor(maxWidth * 1000) / 1000)
  return {
    width: apertureWidth,
    elementCountX: elementCountForAperture(apertureWidth, frequency, bandwidth),
  }
}

function cfarCellLimits(frequency: number, bandwidth: number) {
  const effectiveBandwidth = Math.min(bandwidth, frequency * 0.5)
  const rangeBinSize = SOUND_SPEED / (2 * effectiveBandwidth * 1.1)
  return {
    trainingCells: Math.max(4, Math.round(MAX_CFAR_WINDOW_METERS / rangeBinSize / 2) * 2),
    guardCells: Math.max(1, Math.round(MAX_CFAR_WINDOW_METERS / rangeBinSize)),
  }
}

function colorBound(value: string): number | undefined {
  if (value.trim() === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function ColorScaleControls(props: {
  bounds: ColorBounds
  onChange: (bound: 'min' | 'max', value: string) => void
  unit: string
  disabled?: boolean
}) {
  const min = colorBound(props.bounds.min)
  const max = colorBound(props.bounds.max)
  const invalidRange = min !== undefined && max !== undefined && min >= max
  return (
    <div className="color-scale-controls" aria-label="Color scale limits">
      <label>
        <span>Min {props.unit}</span>
        <input
          type="number"
          value={props.bounds.min}
          placeholder="Auto"
          step="any"
          aria-label={`Color scale minimum (${props.unit})`}
          disabled={props.disabled}
          onChange={(event) => props.onChange('min', event.currentTarget.value)}
        />
      </label>
      <label>
        <span>Max {props.unit}</span>
        <input
          type="number"
          value={props.bounds.max}
          placeholder="Auto"
          step="any"
          aria-label={`Color scale maximum (${props.unit})`}
          disabled={props.disabled}
          onChange={(event) => props.onChange('max', event.currentTarget.value)}
        />
      </label>
      {invalidRange && <span className="color-scale-error" role="alert">Min must be less than max.</span>}
      {props.disabled && <span className="color-scale-hint">Fixed categorical scale</span>}
    </div>
  )
}

const plotConfig: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  scrollZoom: true,
  toImageButtonOptions: { format: 'png', filename: 'sonar-digital-twin', scale: 2 },
}

const commonLayout: Partial<Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#b5c3d4', family: 'Inter, Segoe UI, sans-serif', size: 11 },
  margin: { l: 56, r: 24, t: 18, b: 48 },
  xaxis: { gridcolor: '#263243', zerolinecolor: '#44546a', linecolor: '#354256', automargin: true },
  yaxis: { gridcolor: '#263243', zerolinecolor: '#44546a', linecolor: '#354256', automargin: true },
  hoverlabel: { bgcolor: '#101925', bordercolor: '#324258', font: { color: '#eaf1fa' } },
}

function formatNumber(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

function SliderControl(props: {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit?: string
  description: string
  task: string
  detail?: string
  valueNote?: string
  onChange: (value: number) => void
}) {
  const { label, value, min, max, step, unit = '', description, task, detail, valueNote, onChange } = props
  const [helpOpen, setHelpOpen] = useState(false)
  const shown = Number.isInteger(step) ? formatNumber(value, 0) : formatNumber(value, step < 0.01 ? 3 : 2)
  const limitDigits = Number.isInteger(step) ? 0 : step < 0.1 ? 3 : step < 1 ? 1 : 0
  const helpId = `slider-help-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
  return (
    <div className="control">
      <div className="control-heading">
        <span>{label}</span>
        <span className="control-value">{shown} {unit}</span>
        <div className={`control-help-wrap${helpOpen ? ' is-open' : ''}`}>
          <button
            type="button"
            className="control-help-button"
            aria-label={`Information about ${label}`}
            aria-expanded={helpOpen}
            aria-controls={helpId}
            onClick={() => setHelpOpen((open) => !open)}
          >?</button>
          <div className="control-help-popover" id={helpId} role="region" aria-label={`${label} information`} hidden={!helpOpen}>
            <p>{description}</p>
            {detail && <p className="control-help-detail">{detail}</p>}
            <p className="control-help-task"><strong>Try it:</strong> {task}</p>
          </div>
        </div>
      </div>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={Math.min(value, max)}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
      <span className="range-limits"><span>{formatNumber(min, limitDigits)}</span><span>{formatNumber(max, limitDigits)} {unit}</span></span>
      {valueNote && <span className="control-value-note">{valueNote}</span>}
    </div>
  )
}

function ArrayWeightingControl(props: {
  value: WindowType
  onChange: (value: WindowType) => void
}) {
  const [helpOpen, setHelpOpen] = useState(false)
  const helpId = 'array-weighting-help'
  return (
    <div className="select-control">
      <div className="select-control-heading">
        <label htmlFor="array-weighting">Array weighting</label>
        <span className={`control-help-wrap${helpOpen ? ' is-open' : ''}`}>
          <button
            type="button"
            className="control-help-button"
            aria-label="Information about Array weighting"
            aria-expanded={helpOpen}
            aria-controls={helpId}
            onClick={() => setHelpOpen((open) => !open)}
          >?</button>
          <div className="control-help-popover" id={helpId} role="region" aria-label="Array weighting information" hidden={!helpOpen}>
            <p>Array weighting changes how much each element contributes across the aperture. Rectangular weighting gives all elements equal weight, producing a narrower main lobe but stronger sidelobes. Hamming weighting reduces sidelobes by tapering the array edges, at the cost of a wider main lobe.</p>
            <p className="control-help-task"><strong>Try it:</strong> Switch between Rectangular and Hamming and look for sidelobe effects on the target in the processed data. Enable Strong reflector to make the difference even clearer.</p>
          </div>
        </span>
      </div>
      <select id="array-weighting" value={props.value} onChange={(event) => props.onChange(event.currentTarget.value as WindowType)}>
        <option>Rectangular</option><option>Hamming</option>
      </select>
    </div>
  )
}

function Panel(props: { id: string; title: string; subtitle?: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`panel ${props.className ?? ''}`}>
      <div className="panel-title-row">
        <div>
          <div className="panel-kicker">{props.id}</div>
          <h2>{props.title}</h2>
          {props.subtitle && <p>{props.subtitle}</p>}
        </div>
        <span className="panel-live"><span />LIVE</span>
      </div>
      {props.children}
    </section>
  )
}

function PlotFrame(props: { data: Data[]; layout?: Partial<Layout>; className?: string; uirevision: string }) {
  return (
    <div className={`plot-frame ${props.className ?? ''}`}>
      <Plot
        data={props.data}
        layout={{ ...commonLayout, ...props.layout, autosize: true, uirevision: props.uirevision }}
        config={plotConfig}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}

function MetricTable(props: { title: string; rows: [string, string][] }) {
  return (
    <div className="metric-table-wrap">
      <h3>{props.title}</h3>
      <table className="metric-table">
        <tbody>
          {props.rows.map(([label, value]) => (
            <tr key={label}><th>{label}</th><td>{value}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function App() {
  const [parameters, setParameters] = useState(initialParameters)
  const [linkApertureAndElements, setLinkApertureAndElements] = useState(true)
  const [elementView, setElementView] = useState<ElementView>('Magnitude')
  const [elementColorBounds, setElementColorBounds] = useState<Record<ElementView, ColorBounds>>({
    Magnitude: { ...emptyColorBounds },
    Phase: { ...emptyColorBounds },
    Real: { ...emptyColorBounds },
    Imaginary: { ...emptyColorBounds },
  })
  const [resultTab, setResultTab] = useState<ResultMode>('matched')
  const [resultCoordinatesHelpOpen, setResultCoordinatesHelpOpen] = useState(false)
  const [resultColorBounds, setResultColorBounds] = useState<Record<ResultMode, ColorBounds>>({
    beam: { ...emptyColorBounds },
    matched: { ...emptyColorBounds },
    cfar: { ...emptyColorBounds },
    detections: { ...emptyColorBounds },
  })
  const [resultCoordinates, setResultCoordinates] = useState<'polar' | 'xy'>('polar')
  const simulationBase = useMemo(
    () => simulate(parameters),
    [
      parameters.width,
      parameters.elementCountX,
      parameters.frequency,
      parameters.bandwidth,
      parameters.pulseDuration,
      parameters.prr,
      parameters.fanAngle,
      parameters.sourceLevel,
      parameters.reflectorEnabled,
      parameters.reflectorStrengthDb,
      parameters.window,
    ],
  )
  const cfarResults = useMemo(
    () => calculateCfar(simulationBase, parameters),
    [simulationBase, parameters.trainingCells, parameters.guardCells],
  )
  const detectionResults = useMemo(
    () => calculateDetections(cfarResults, parameters.threshold),
    [cfarResults, parameters.threshold],
  )
  const simulation: Simulation = useMemo(
    () => ({ ...simulationBase, ...cfarResults, ...detectionResults }),
    [simulationBase, cfarResults, detectionResults],
  )

  const update = <K extends keyof Parameters>(key: K, value: Parameters[K]) => {
    setParameters((current) => ({ ...current, [key]: value }))
  }
  const bandEdgeFrequency = parameters.frequency + parameters.bandwidth / 2
  const linkedElementSpacing = halfWavelengthAtBandEdge(parameters.frequency, parameters.bandwidth)
  const linkedElementCountMin = elementCountForAperture(
    MIN_APERTURE_WIDTH,
    parameters.frequency,
    parameters.bandwidth,
  )
  const linkedElementCountMax = Math.min(
    MAX_ARRAY_ELEMENTS,
    Math.floor(MAX_APERTURE_WIDTH / linkedElementSpacing) + 1,
  )

  const updateElementColorBound = (bound: 'min' | 'max', value: string) => {
    setElementColorBounds((current) => ({
      ...current,
      [elementView]: { ...current[elementView], [bound]: value },
    }))
  }
  const updateResultColorBound = (bound: 'min' | 'max', value: string) => {
    setResultColorBounds((current) => ({
      ...current,
      [resultTab]: { ...current[resultTab], [bound]: value },
    }))
  }

  const fanHalfAngle = parameters.fanAngle / 2
  const sceneFanAngles = Array.from({ length: Math.round(parameters.fanAngle) + 1 }, (_, index) => (
    (-fanHalfAngle + parameters.fanAngle * index / Math.round(parameters.fanAngle)) * Math.PI / 180
  ))
  const sceneData: Data[] = [
    {
      type: 'scatter',
      mode: 'lines',
      x: [0, ...sceneFanAngles.map((angle) => 110 * Math.sin(angle)), 0],
      y: [0, ...sceneFanAngles.map((angle) => 110 * Math.cos(angle)), 0],
      line: { color: 'rgba(83, 217, 193, 0.45)', width: 1 },
      fill: 'toself',
      fillcolor: 'rgba(83, 217, 193, 0.10)',
      name: `Receive scan fan (${parameters.fanAngle}°)`,
      hoverinfo: 'skip',
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: [...simulation.targetOutline.map((point) => point.x), simulation.targetOutline[0].x],
      y: [...simulation.targetOutline.map((point) => point.y), simulation.targetOutline[0].y],
      line: { color: '#ffb454', width: 2 },
      name: 'Target outline',
      hoverinfo: 'skip',
    },
    {
      type: 'scatter',
      mode: 'markers',
      x: simulation.scatterers.map((point) => point.x),
      y: simulation.scatterers.map((point) => point.y),
      marker: { color: '#ffd08a', size: 3 },
      name: 'Visible surface scatterers',
      hovertemplate: 'Cross-range %{x:.2f} m<br>Range %{y:.2f} m<extra></extra>',
    },
    ...(simulation.reflector ? [{
      type: 'scatter' as const,
      mode: 'markers' as const,
      x: [simulation.reflector.x],
      y: [simulation.reflector.y],
      marker: { color: '#f35b68', size: 12, symbol: 'star' as const, line: { color: '#ffd0d4', width: 1 } },
      name: 'Strong reflector (non-blocking)',
      hovertemplate: 'Strong reflector<br>Cross-range %{x:.2f} m<br>Range %{y:.2f} m<extra></extra>',
    }] : []),
    {
      type: 'scatter',
      mode: 'markers',
      x: [0],
      y: [0],
      marker: { color: '#53d9c1', size: 12, symbol: 'diamond', line: { color: '#c4fff2', width: 1 } },
      name: 'Sonar',
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: [-parameters.width / 2, parameters.width / 2],
      y: [0, 0],
      line: { color: '#53d9c1', width: 1.5 },
      name: 'Linear array aperture',
    },
  ]
  const sceneLayout: Partial<Layout> = {
    xaxis: { ...commonLayout.xaxis, title: { text: 'Cross-range (m)' }, range: [-110, 110] },
    yaxis: { ...commonLayout.yaxis, title: { text: 'Range (m)' }, range: [-110, 110] },
    showlegend: true,
    legend: {
      orientation: 'v',
      x: 0,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: 'rgba(13,13,24,0.72)',
    },
    margin: { l: 54, r: 18, t: 30, b: 48 },
  }

  const elementPlot = useMemo(() => plotMatrix(simulationBase, elementView), [simulationBase, elementView])
  const elementBounds = elementColorBounds[elementView]
  const elementMin = colorBound(elementBounds.min)
  const elementMax = colorBound(elementBounds.max)
  const invalidElementRange = elementMin !== undefined && elementMax !== undefined && elementMin >= elementMax
  const elementData: Data[] = [{
    type: 'heatmap',
    z: elementPlot.z,
    x: elementPlot.x,
    y: elementPlot.y,
    ...(!invalidElementRange ? { zmin: elementMin, zmax: elementMax } : {}),
    colorscale: elementView === 'Phase' ? 'Twilight' : 'Viridis',
    colorbar: { title: { text: elementPlot.colorbarTitle }, thickness: 12 },
    hovertemplate: elementView === 'Phase'
      ? 'Range %{x:.2f} m<br>Element %{y}<br>Phase %{z:.4g} rad<extra></extra>'
      : 'Range %{x:.2f} m<br>Element %{y}<br>Level %{z:.2f} dB<extra></extra>',
  }]
  const elementLayout: Partial<Layout> = {
    xaxis: { ...commonLayout.xaxis, title: { text: 'Fast time (range equivalent, m)' } },
    yaxis: { ...commonLayout.yaxis, title: { text: 'Representative element index' } },
    margin: { l: 64, r: 68, t: 12, b: 48 },
  }

  const resultModes = [
    { key: 'beam', label: 'Beamformed' },
    { key: 'matched', label: 'Matched filter' },
    { key: 'cfar', label: 'CA-CFAR' },
    { key: 'detections', label: 'Detections' },
  ] as const
  const resultMatrixData = useMemo(() => resultMatrix(simulation, resultTab), [simulation, resultTab])
  const cartesianResult = useMemo(
    () => cartesianMatrix(resultMatrixData, simulation.rangeAxis, simulation.angleAxis, resultTab === 'detections'),
    [resultMatrixData, simulation.rangeAxis, simulation.angleAxis, resultTab],
  )
  const cartesianDetectionPoints = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    const values: number[] = []
    if (resultTab === 'detections') {
      for (let angleIndex = 0; angleIndex < simulation.angleAxis.length; angleIndex += 1) {
        const angle = simulation.angleAxis[angleIndex] * Math.PI / 180
        for (let rangeIndex = 0; rangeIndex < simulation.rangeAxis.length; rangeIndex += 1) {
          const value = resultMatrixData[angleIndex][rangeIndex]
          if (value === 0) continue
          const range = simulation.rangeAxis[rangeIndex]
          x.push(range * Math.sin(angle))
          y.push(range * Math.cos(angle))
          values.push(value)
        }
      }
    }
    return { x, y, values }
  }, [resultMatrixData, resultTab, simulation.angleAxis, simulation.rangeAxis])
  const isCartesian = resultCoordinates === 'xy'
  const resultValueLabel = resultTab === 'cfar' ? 'Power ratio' : 'Level'
  const resultBounds = resultColorBounds[resultTab]
  const resultMin = colorBound(resultBounds.min)
  const resultMax = colorBound(resultBounds.max)
  const invalidResultRange = resultMin !== undefined && resultMax !== undefined && resultMin >= resultMax
  const resultScale = resultTab === 'detections'
    ? [[0, '#f35b68'], [0.49, '#f35b68'], [0.5, '#182331'], [0.51, '#182331'], [1, '#49df99']] as [number, string][]
    : resultTab === 'cfar' ? 'Turbo' : 'Viridis'
  const resultData: Data[] = [{
    type: 'heatmap',
    z: isCartesian ? cartesianResult.z : resultMatrixData,
    x: isCartesian ? cartesianResult.axis : simulation.rangeAxis,
    y: isCartesian ? cartesianResult.axis : simulation.angleAxis,
    colorscale: resultScale,
    zmin: resultTab === 'detections' ? -1 : undefined,
    zmax: resultTab === 'detections' ? 1 : undefined,
    ...(resultTab !== 'detections' && !invalidResultRange
      ? { zmin: resultMin, zmax: resultMax }
      : {}),
    colorbar: {
    title: { text: resultTab === 'cfar' ? 'Power ratio (dB)' : resultTab === 'detections' ? 'False / none / true' : 'Relative level (dB)' },
      thickness: 12,
    },
    hovertemplate: isCartesian
    ? `Cross-range %{x:.2f} m<br>Down-range %{y:.2f} m<br>${resultTab === 'detections' ? 'Class' : resultValueLabel} %{z:.2f}${resultTab === 'detections' ? '' : ' dB'}<extra></extra>`
    : `Range %{x:.2f} m<br>Bearing %{y:.1f}°<br>${resultTab === 'detections' ? 'Class' : resultValueLabel} %{z:.2f}${resultTab === 'detections' ? '' : ' dB'}<extra></extra>`,
  }, ...(isCartesian && resultTab === 'detections' ? [{
    type: 'scattergl' as const,
    mode: 'markers' as const,
    x: cartesianDetectionPoints.x,
    y: cartesianDetectionPoints.y,
    marker: {
      color: cartesianDetectionPoints.values,
      colorscale: resultScale,
      cmin: -1,
      cmax: 1,
      size: 6,
      showscale: false,
    },
    name: 'Detections',
    hovertemplate: 'Cross-range %{x:.2f} m<br>Down-range %{y:.2f} m<extra></extra>',
  }] : [])]
  const resultLayout: Partial<Layout> = {
    xaxis: {
      ...commonLayout.xaxis,
      title: { text: isCartesian ? 'Cross-range x (m)' : 'Range (m)' },
    },
    yaxis: {
      ...commonLayout.yaxis,
      title: { text: isCartesian ? 'Down-range y (m)' : 'Bearing (deg)' },
      ...(isCartesian
        ? { scaleanchor: 'x', scaleratio: 1 }
        : {}),
    },
    margin: { l: 58, r: 80, t: 12, b: 48 },
  }

  const geometryData: Data[] = [
    {
      type: 'scattergl',
      mode: 'markers',
      x: simulation.elementX,
      y: simulation.elementZ,
      marker: { color: '#53d9c1', size: 5, opacity: 0.85 },
      name: 'Array elements',
      hovertemplate: 'x %{x:.3f} m<br>z %{y:.3f} m<extra></extra>',
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: [-parameters.width / 2, parameters.width / 2],
      y: [0, 0],
      line: { color: '#7d8ba0', dash: 'dot', width: 1 },
      name: 'Aperture',
    },
  ]
  const geometryLayout: Partial<Layout> = {
    xaxis: { ...commonLayout.xaxis, title: { text: 'Aperture width x (m)' } },
    yaxis: { ...commonLayout.yaxis, title: { text: 'Element position z (m)' }, scaleanchor: 'x', scaleratio: 1 },
    showlegend: false,
    margin: { l: 56, r: 18, t: 12, b: 48 },
  }

  const parameterRows: [string, string][] = [
    ['Aperture width', `${formatNumber(parameters.width)} m`],
    ['Horizontal elements', formatNumber(simulation.nx, 0)],
    ['Total elements', formatNumber(simulation.totalElements, 0)],
    ['Carrier frequency', `${formatNumber(parameters.frequency / 1000, 1)} kHz`],
    ['Bandwidth', `${formatNumber(parameters.bandwidth / 1000, 1)} kHz`],
    ['Sampling rate (Fs)', `${formatNumber(simulation.sampleRate / 1000, 2)} kHz`],
    ['Wavelength', `${formatNumber(simulation.wavelength * 100, 2)} cm`],
    ['Range resolution (c/2B)', `${formatNumber(simulation.rangeResolution, 3)} m`],
    ['Horizontal beamwidth', `${formatNumber(simulation.beamwidthH, 2)}°`],
    ['Receive beams', formatNumber(simulation.rows, 0)],
    ['Beam spacing', `${formatNumber(simulation.beamSpacing, 3)}°${simulation.beamCountLimited ? ' (limited)' : ''}`],
    ['Vertical beamwidth', `${formatNumber(simulation.beamwidthV, 2)}°`],
    ['Directivity factor', formatNumber(simulation.directivity, 0)],
    ['Directivity index', `${formatNumber(simulation.directivityIndex, 2)} dB`],
    ['Source level', `${formatNumber(parameters.sourceLevel, 0)} dB re 1 µPa`],
    ['Pulse duration', `${formatNumber(parameters.pulseDuration, 1)} ms`],
    ['Receive scan fan', `${formatNumber(parameters.fanAngle, 0)}°`],
    ['Maximum unambiguous range', `${formatNumber(simulation.maxUnambiguousRange, 1)} m`],
    ['Beamforming window', parameters.window],
    ['Training cells', formatNumber(parameters.trainingCells, 0)],
    ['Guard cells', formatNumber(parameters.guardCells, 0)],
    ['Detection threshold', `${formatNumber(parameters.threshold, 1)}×`],
    ['Estimated element SNR', `${formatNumber(simulation.estimatedSnr, 1)} dB`],
    ['Target SNR (peak)', simulation.targetSnrDb === undefined ? '—' : `${formatNumber(simulation.targetSnrDb, 1)} dB`],
    ['Detection count', formatNumber(simulation.detectionCount, 0)],
    ['False-alarm count', formatNumber(simulation.falseAlarmCount, 0)],
    ['False-alarm rate', `${(simulation.falseAlarmRate * 100).toFixed(3)}%`],
  ]
  const bandwidthMax = parameters.frequency / 2
  const bandwidthMin = Math.min(500, bandwidthMax)
  const rangeBinSize = simulation.rangeAxis[1] - simulation.rangeAxis[0]
  const cfarLimits = cfarCellLimits(parameters.frequency, parameters.bandwidth)
  const trainingCellsPerSide = Math.floor(parameters.trainingCells / 2)

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-logo" aria-label="USN">USN</div>
          <div><div className="brand-eyebrow">UNIVERSITETET I SØRØST-NORGE</div><h1>Sonar Digital Twin</h1></div>
        </div>
        <div className="system-state"><span className="state-dot" /> SIMULATION ONLINE <span className="state-divider" /> c = 1500 m/s</div>
      </header>

      <section className="summary-strip">
        <div><span className="summary-label">ARRAY</span><strong>{formatNumber(simulation.totalElements, 0)}</strong><small>elements</small></div>
        <div><span className="summary-label">BEAMWIDTH</span><strong>{formatNumber(simulation.beamwidthH, 2)}°</strong><small>horizontal · −3 dB</small></div>
        <div><span className="summary-label">RECEIVE BEAMS</span><strong>{formatNumber(simulation.rows, 0)}</strong><small>{formatNumber(simulation.beamSpacing, 3)}° spacing{simulation.beamCountLimited ? ' · limited' : ''}</small></div>
        <div><span className="summary-label">DIRECTIVITY</span><strong>{formatNumber(simulation.directivityIndex, 1)} <small>dB</small></strong><small>estimated DI</small></div>
        <div><span className="summary-label">FALSE ALARMS</span><strong>{formatNumber(simulation.falseAlarmCount, 0)}</strong><small>{(simulation.falseAlarmRate * 100).toFixed(3)}% of coverage</small></div>
        <div><span className="summary-label">TARGET SNR</span><strong>{simulation.targetSnrDb === undefined ? '—' : `${formatNumber(simulation.targetSnrDb, 1)} dB`}</strong><small>peak target / median background power</small></div>
      </section>

      <div className="dashboard-grid">
        <Panel id="01 / SCENE" title="Ground truth" subtitle={`220 × 220 m scene · ${parameters.fanAngle}° receive scan fan · fixed five-point star at 80 m boresight${simulation.reflector ? ' · reflector at 55 m' : ''}`} className="scene-panel">
          <PlotFrame data={sceneData} layout={sceneLayout} uirevision="scene" />
          <div className="scene-caption"><span><i className="legend-sonar" />Sonar at origin</span><span><i className="legend-fan" />Receive scan fan ({parameters.fanAngle}°)</span><span><i className="legend-target" />Target outline; far-side surface is shadowed</span></div>
        </Panel>

        <Panel id="02 / CHANNELS" title="Element data" subtitle={`${simulation.simElementCount} representative channels · exact physical array count shown in metrics`} className="element-panel">
          <div className="plot-toolbar">
            <span>Complex baseband echo + thermal noise</span>
            <select value={elementView} onChange={(event) => setElementView(event.currentTarget.value as ElementView)} aria-label="Element data view">
              <option>Magnitude</option><option>Phase</option><option>Real</option><option>Imaginary</option>
            </select>
          </div>
          <ColorScaleControls
            bounds={elementBounds}
            onChange={updateElementColorBound}
            unit={elementView === 'Phase' ? 'rad' : 'dB'}
          />
          <PlotFrame data={elementData} layout={elementLayout} uirevision="element-data" />
        </Panel>

        <Panel id="03 / PROCESSING" title="Processing chain" subtitle="Delay-and-sum → matched filter → CA-CFAR → threshold" className="results-panel">
          <div className="tab-row" role="tablist">
            {resultModes.map((tab) => (
              <button
                type="button"
                key={tab.key}
                className={resultTab === tab.key ? 'active' : ''}
                onClick={() => setResultTab(tab.key)}
                role="tab"
                aria-selected={resultTab === tab.key}
              >{tab.label}</button>
            ))}
          </div>
          <div className="result-toolbar">
            <span>Coordinates</span>
            <div className="result-coordinate-controls">
              <div className="result-view-toggle" role="group" aria-label="Processed data coordinates">
                <button
                  type="button"
                  className={!isCartesian ? 'active' : ''}
                  aria-pressed={!isCartesian}
                  onClick={() => setResultCoordinates('polar')}
                >θ–r</button>
                <button
                  type="button"
                  className={isCartesian ? 'active' : ''}
                  aria-pressed={isCartesian}
                  onClick={() => setResultCoordinates('xy')}
                >XY</button>
              </div>
              <div className={`control-help-wrap${resultCoordinatesHelpOpen ? ' is-open' : ''}`}>
                <button
                  type="button"
                  className="control-help-button"
                  aria-label="Information about coordinate views"
                  aria-expanded={resultCoordinatesHelpOpen}
                  aria-controls="result-coordinate-help"
                  onClick={() => setResultCoordinatesHelpOpen((open) => !open)}
                >?</button>
                <div
                  className="control-help-popover"
                  id="result-coordinate-help"
                  role="region"
                  aria-label="Coordinate view information"
                  hidden={!resultCoordinatesHelpOpen}
                >
                  <p className="control-help-task"><strong>Try it:</strong> The theta-r view displays the acquired data, where each horizontal line corresponds to an individual beam. The xy view transforms the data into a spatial representation, providing a more intuitive and realistic depiction of the surrounding environment.</p>
                </div>
              </div>
            </div>
          </div>
          <ColorScaleControls
            bounds={resultBounds}
            onChange={updateResultColorBound}
            unit="dB"
            disabled={resultTab === 'detections'}
          />
          <PlotFrame data={resultData} layout={resultLayout} uirevision="processing-results" />
          <div className="result-legend">
            {resultTab === 'detections'
              ? <><span><i className="legend-true" />True detection</span><span><i className="legend-false" />False alarm</span></>
              : <span>Color indicates {resultTab === 'cfar' ? 'CUT / local noise power ratio (dB)' : 'beamformed signal level (dB)'} · zoom and pan enabled</span>}
          </div>
        </Panel>

        <Panel id="04 / ARRAY" title="Sonar geometry" subtitle="Single-row linear array across the physical aperture" className="geometry-panel">
          <PlotFrame data={geometryData} layout={geometryLayout} uirevision="array-geometry" />
          <div className="geometry-caption">
            <span>Single-row linear array · One element in height</span>
            <span>Actual grid: {simulation.nx} × {simulation.ny}</span>
          </div>
        </Panel>

        <Panel id="05 / CONTROLS" title="Parameters & metrics" subtitle="Beamforming, pulse, and detection settings update the simulation interactively." className="controls-panel">
          <div className="control-metrics-grid">
            <div className="controls-column">
              <div className="control-section">
                <div className="control-section-heading">
                  <h3><span className="section-index">A</span> Beamforming</h3>
                  <label className="array-link-control">
                    <input
                      type="checkbox"
                      checked={linkApertureAndElements}
                      onChange={(event) => {
                        const linked = event.currentTarget.checked
                        setLinkApertureAndElements(linked)
                        if (linked) {
                          setParameters((current) => {
                            const dimensions = linkedArrayForAperture(
                              current.width,
                              current.frequency,
                              current.bandwidth,
                            )
                            return { ...current, ...dimensions }
                          })
                        }
                      }}
                    />
                    <span>Link aperture and elements</span>
                  </label>
                </div>
                <div className="select-grid">
                  <SliderControl
                    label="Aperture width"
                    value={parameters.width}
                    min={MIN_APERTURE_WIDTH}
                    max={MAX_APERTURE_WIDTH}
                    step={linkApertureAndElements ? 0.001 : 0.1}
                    unit="m"
                    description="The physical width of the array aperture. When linked, changing aperture width also updates the element count to keep element spacing at or below half a wavelength at the upper band edge. This might put some limitation on the choice of array aperture width."
                    task="Increase the width and watch the horizontal beamwidth in the summary and notice the changes in the processed data."
                    onChange={(value) => setParameters((current) => ({
                      ...current,
                      ...(linkApertureAndElements
                        ? linkedArrayForAperture(value, current.frequency, current.bandwidth)
                        : { width: value }),
                    }))}
                  />
                  <SliderControl
                    label="Elements"
                    value={parameters.elementCountX}
                    min={linkApertureAndElements ? linkedElementCountMin : 1}
                    max={linkApertureAndElements ? linkedElementCountMax : MAX_ARRAY_ELEMENTS}
                    step={1}
                    description="The number of transducer elements across the array. When linked, changing element count also updates aperture width to use half-wavelength spacing at the upper band edge."
                    detail={`Current element spacing is ${simulation.nx > 1 ? `${formatNumber(parameters.width / (simulation.nx - 1) * 100, 2)} cm` : 'n/a'}. The half-wavelength limit at the band edge (${formatNumber(bandEdgeFrequency / 1000, 1)} kHz) is ${formatNumber(linkedElementSpacing * 100, 2)} cm.`}
                    task="Try unlinking the aperture and elements, increase the Recieve scan fan to 180 degrees, and observe the changes in the processed data when not meeting the half-wavelength criterium."
                    onChange={(value) => setParameters((current) => ({
                      ...current,
                      ...(linkApertureAndElements
                        ? {
                          elementCountX: value,
                          width: (value - 1) * halfWavelengthAtBandEdge(current.frequency, current.bandwidth),
                        }
                        : { elementCountX: value }),
                    }))}
                  />
                  <SliderControl
                    label="Receive scan fan"
                    value={parameters.fanAngle}
                    min={30}
                    max={360}
                    step={30}
                    unit="°"
                    description="The total angular sector covered by the receive beamformer, centered on the forward direction. The same sector is shown in the ground-truth scene."
                    task="Widen the scan fan and compare the coverage in the scene and processed-data views."
                    onChange={(value) => update('fanAngle', value)}
                  />
                  <ArrayWeightingControl value={parameters.window} onChange={(value) => update('window', value)} />
                </div>
              </div>
              <div className="control-section">
                <h3><span className="section-index">B</span> Pulse parameters</h3>
                <div className="slider-grid">
                  <SliderControl label="Carrier frequency" value={parameters.frequency / 1000} min={10} max={500} step={1} unit="kHz" description="The center frequency of the transmitted pulse. Higher frequencies shorten the wavelength and increase absorption in water." task="Increase the frequency and observe its effect on the SNR. Relate frequency to aperture size and number of elements, and consider how changes in aperture width influence image quality." onChange={(value) => {
                    const frequency = value * 1000
                    setParameters((current) => {
                      const bandwidth = Math.min(current.bandwidth, frequency / 2)
                      const limits = cfarCellLimits(frequency, bandwidth)
                      return {
                        ...current,
                        frequency,
                        bandwidth,
                        trainingCells: Math.min(current.trainingCells, limits.trainingCells),
                        guardCells: Math.min(current.guardCells, limits.guardCells),
                        ...(linkApertureAndElements
                          ? linkedArrayForAperture(current.width, frequency, bandwidth)
                          : {}),
                      }
                    })
                  }} />
                  <SliderControl
                    label="Bandwidth"
                    value={parameters.bandwidth / 1000}
                    min={bandwidthMin / 1000}
                    max={bandwidthMax / 1000}
                    step={0.5}
                    unit="kHz"
                    description="The transmitted signal bandwidth. Greater bandwidth improves range resolution, approximately c/(2B)."
                    detail={`Maximum bandwidth is 50% of the carrier frequency (${formatNumber(bandwidthMax / 1000, 1)} kHz).`}
                    task="Double the bandwidth and compare the reported range resolution."
                    onChange={(value) => {
                      const bandwidth = value * 1000
                      setParameters((current) => {
                        const limits = cfarCellLimits(current.frequency, bandwidth)
                        return {
                          ...current,
                          trainingCells: Math.min(current.trainingCells, limits.trainingCells),
                          guardCells: Math.min(current.guardCells, limits.guardCells),
                          ...(linkApertureAndElements
                            ? { bandwidth, ...linkedArrayForAperture(current.width, current.frequency, bandwidth) }
                            : { bandwidth }),
                        }
                      })
                    }}
                  />
                  <SliderControl label="Pulse duration" value={parameters.pulseDuration} min={0.1} max={50} step={0.1} unit="ms" description="Pulse duration in milliseconds. Longer pulses provide higher matched-filter gain and improved SNR." task="Double the pulse duration and check that estimated SNR rises by about 3 dB." onChange={(value) => update('pulseDuration', value)} />
                  <SliderControl label="Source level" value={parameters.sourceLevel} min={150} max={250} step={1} unit="dB" description="The transmitted sound pressure level. Increasing source level raises the received signal level and can improve SNR." task="Increase source level by 10 dB and compare the target SNR." onChange={(value) => update('sourceLevel', value)} />
                </div>
              </div>
              <div className="control-section">
                <h3><span className="section-index">C</span> Detection</h3>
                <div className="slider-grid">
                  <SliderControl
                    label="CFAR training cells"
                    value={parameters.trainingCells}
                    min={4}
                    max={cfarLimits.trainingCells}
                    step={2}
                    unit="cells"
                    description="The total number of reference cells, split equally between the two sides of the cell under test."
                    detail={`${trainingCellsPerSide} cells per side; ${formatNumber(trainingCellsPerSide * rangeBinSize, 2)} m per side.`}
                    valueNote={`${parameters.trainingCells} cells total · ${formatNumber(parameters.trainingCells * rangeBinSize, 2)} m total (${trainingCellsPerSide} cells / ${formatNumber(trainingCellsPerSide * rangeBinSize, 2)} m per side) · max ${cfarLimits.trainingCells} cells (${formatNumber(cfarLimits.trainingCells * rangeBinSize, 2)} m total)`}
                    task="Add the strong reflector and change the training-cell count. Observe how this influences the CA-CFAR and Detection output."
                    onChange={(value) => update('trainingCells', value)}
                  />
                  <SliderControl
                    label="CFAR guard cells"
                    value={parameters.guardCells}
                    min={1}
                    max={cfarLimits.guardCells}
                    step={1}
                    unit="cells/side"
                    description="Guard cells are excluded from the noise estimate on each side of the cell under test."
                    detail={`${parameters.guardCells} cells (${formatNumber(parameters.guardCells * rangeBinSize, 2)} m) on each side.`}
                    valueNote={`${parameters.guardCells} cells/side · ${formatNumber(parameters.guardCells * rangeBinSize, 2)} m/side (${parameters.guardCells * 2} cells · ${formatNumber(parameters.guardCells * 2 * rangeBinSize, 2)} m total) · max ${cfarLimits.guardCells} cells/side (${formatNumber(cfarLimits.guardCells * rangeBinSize, 2)} m/side)`}
                    task="Add the strong reflector and change the guard-cell count. Observe how this influences the CA-CFAR and Detection output."
                    onChange={(value) => update('guardCells', value)}
                  />
                  <SliderControl label="Detection threshold" value={parameters.threshold} min={2} max={40} step={0.5} unit="×" description="The required ratio of cell-under-test power to estimated local noise for a detection. A higher threshold is more conservative and reduces false alarms." task="Raise the threshold and compare the detection and false-alarm counts." onChange={(value) => update('threshold', value)} />
                </div>
              </div>
              <div className="control-section">
                <div className="control-section-heading">
                  <h3><span className="section-index">D</span> Strong reflector</h3>
                  <label className="array-link-control">
                    <input
                      type="checkbox"
                      checked={parameters.reflectorEnabled}
                      onChange={(event) => update('reflectorEnabled', event.currentTarget.checked)}
                    />
                    <span>Enable reflector</span>
                  </label>
                </div>
                {parameters.reflectorEnabled && (
                  <div className="slider-grid">
                    <SliderControl
                      label="Reflector strength"
                      value={parameters.reflectorStrengthDb}
                      min={0}
                      max={40}
                      step={1}
                      unit="dB"
                      description="Adds a strong point echo at 55 m on the sonar's centerline, 25 m in front of the target center. The reflector adds an echo but does not block or shadow the target."
                      detail="Strength is relative to the echo amplitude of one target surface scatterer."
                      task="Enable the reflector, then compare the matched-filter and detection plots with the Hamming and Rectangular array windows."
                      onChange={(value) => update('reflectorStrengthDb', value)}
                    />
                  </div>
                )}
              </div>
              <div className="model-note"><span className="note-icon">i</span><span>Aperture width and element count can be linked; the array is limited to {MAX_ARRAY_ELEMENTS} elements. When linked, the element spacing is kept at or below λ/2 at the upper band edge. If 192 elements are insufficient, aperture width is reduced to preserve the spacing limit. When unlinked, frequency changes leave both array dimensions unchanged. Thermal-noise level rises with frequency; absorption uses a Thorp approximation. Sector-wide transmission illuminates the full target, while the receive beamformer scans the selected fan using the full array. Beam spacing targets half the horizontal beamwidth. Signal synthesis uses up to 96 representative element channels; the angle–range grid is limited to 250,000 cells for interactive performance.{simulation.beamCountLimited ? ` The current ${formatNumber(simulation.rows, 0)} beams use ${formatNumber(simulation.beamSpacing, 3)}° spacing, wider than half the beamwidth.` : ''}</span></div>
            </div>
            <div className="metrics-column">
              <MetricTable title="Parameter table" rows={parameterRows} />
            </div>
          </div>
        </Panel>
      </div>

      <footer>
        <span>ACTIVE SONAR / NUMERICAL SIMULATION</span>
        <span>Complex baseband · spherical spreading · frequency-dependent absorption</span>
      </footer>
    </main>
  )
}
