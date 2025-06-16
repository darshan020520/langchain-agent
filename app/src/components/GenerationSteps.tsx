const GenerationSteps = ({ steps, done }: { steps: Step[]; done: boolean }) => {
  return (
    <div className="mt-5">
      <p className="text-xs text-gray-500">Generation steps:</p>

      <div className="flex flex-col gap-1 mt-1">
        {steps.map((step, i) => (
          <div
            key={i}
            className="flex items-center gap-1 text-xs px-1 py-[1px] bg-gray-800 rounded text-white"
          >
            <p>{step.name}</p>
            {step.result && (
              <p className="text-gray-400">
                {typeof step.result === 'string' 
                  ? step.result 
                  : JSON.stringify(step.result)}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}; 