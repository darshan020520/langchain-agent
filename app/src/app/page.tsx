"use client";

import Output from "@/components/Output";
import TextArea from "@/components/TextArea";
import { type ChatOutput } from "@/types";
import { useState, useCallback } from "react";

export default function Home() {
  const [outputs, setOutputs] = useState<ChatOutput[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const updateOutput = useCallback((index: number, update: Partial<ChatOutput>) => {
    setOutputs(prev => {
      const newOutputs = [...prev];
      if (index >= 0 && index < newOutputs.length) {
        newOutputs[index] = { ...newOutputs[index], ...update };
      }
      return newOutputs;
    });
  }, []);

  return (
    <div
      className={`container pt-10 pb-32 min-h-screen ${
        outputs.length === 0 && "flex items-center justify-center"
      }`}
    >
      <div className="w-full">
        {outputs.length === 0 && (
          <h1 className="text-4xl text-center mb-5">
            What do you want to know?
          </h1>
        )}

        <TextArea
          setIsGenerating={setIsGenerating}
          isGenerating={isGenerating}
          outputs={outputs}
          setOutputs={setOutputs}
          updateOutput={updateOutput}
        />

        {outputs.map((output, i) => {
          return <Output key={i} output={output} />;
        })}
      </div>
    </div>
  );
}
