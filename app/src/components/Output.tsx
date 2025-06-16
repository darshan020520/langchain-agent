import React from 'react';
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { Step, type ChatOutput } from "@/types";
import { useEffect, useState } from "react";

const Output = ({ output }: { output: ChatOutput }) => {
  const detailsHidden = !!output.result?.answer;
  const [isLoading, setIsLoading] = useState(false);
  
  // Debug logging
  useEffect(() => {
    console.log("Output component received:", output);
    console.log("Output details:", {
      hasResult: !!output.result,
      hasAnswer: !!output.result?.answer,
      hasSteps: !!output.steps?.length,
      hasTools: !!output.result?.tools_used?.length
    });
    setIsLoading(!output.result?.answer);
  }, [output]);

  return (
    <div className="border-t border-gray-700 py-10 first-of-type:pt-0 first-of-type:border-t-0">
      <p className="text-3xl">{output.question}</p>

      {/* Steps */}
      {output.steps && output.steps.length > 0 && (
        <div className="mt-5">
          <p className="text-xs text-gray-500">Generation steps:</p>
          <div className="flex flex-col gap-1 mt-1">
            {output.steps.map((step, i) => (
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
      )}

      {/* Output */}
      <div
        className="mt-5 prose dark:prose-invert min-w-full prose-pre:whitespace-pre-wrap"
        style={{
          overflowWrap: "anywhere",
        }}
      >
        {output.result?.answer ? (
          <MarkdownRenderer content={output.result.answer} />
        ) : (
          <p className="text-gray-500">
            {isLoading ? "Generating response..." : "No response yet"}
          </p>
        )}
      </div>

      {/* Tools */}
      {output.result?.tools_used && output.result.tools_used.length > 0 && (
        <div className="flex items-baseline mt-5 gap-1">
          <p className="text-xs text-gray-500">Tools used:</p>
          <div className="flex flex-wrap items-center gap-1">
            {output.result.tools_used.map((tool, i) => (
              <p
                key={i}
                className="text-xs px-1 py-[1px] bg-gray-800 rounded text-white"
              >
                {tool}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const GenerationSteps = ({ steps, done }: { steps: Step[]; done: boolean }) => {
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    if (done) setHidden(true);
  }, [done]);

  // Debug logging
  useEffect(() => {
    console.log("GenerationSteps received:", { steps, done });
  }, [steps, done]);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="border border-gray-700 rounded mt-5 p-3 flex flex-col">
      <button
        className="w-full text-left flex items-center justify-between"
        onClick={() => setHidden(!hidden)}
      >
        Steps {hidden ? <ChevronDown /> : <ChevronUp />}
      </button>

      {!hidden && (
        <div className="space-y-2">
          {steps.map((step, index) => (
            <div key={index} className="p-2 bg-gray-800 rounded">
              <div className="flex items-center gap-2">
                <p className="text-sm font-medium text-white">{step.name}</p>
              </div>
              <div className="mt-1">
                <p className="text-sm text-gray-300">
                  Result: {typeof step.result === 'object' ? JSON.stringify(step.result) : step.result}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const ChevronDown = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="w-4 h-4"
  >
    <path d="m6 9 6 6 6-6" />
  </svg>
);

const ChevronUp = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="w-4 h-4"
  >
    <path d="m18 15-6-6-6 6" />
  </svg>
);

export default Output;
