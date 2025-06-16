"use client";

import { useEffect, useRef, useState } from "react";
import { ChatOutput } from "@/types";

const TextArea = ({
  setIsGenerating,
  isGenerating,
  setOutputs,
  outputs,
  updateOutput,
}: {
  setIsGenerating: React.Dispatch<React.SetStateAction<boolean>>;
  isGenerating: boolean;
  setOutputs: React.Dispatch<React.SetStateAction<ChatOutput[]>>;
  outputs: ChatOutput[];
  updateOutput: (index: number, update: Partial<ChatOutput>) => void;
}) => {
  const [text, setText] = useState("");
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    try {
      setIsGenerating(true);
      setOutputs(prev => [...prev, { question: text, steps: [], result: null }]);
      setText("");
      console.log("[DEBUG] Starting request with input:", text);

      const response = await fetch(`http://localhost:8000/invoke?content=${encodeURIComponent(text)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("[DEBUG] Stream complete");
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        console.log("[DEBUG] Received chunk:", buffer);

        // Process complete messages
        const messages = buffer.split('\n');
        buffer = messages.pop() || ''; // Keep the last incomplete message in the buffer

        for (const message of messages) {
          if (message.trim() === '') continue;
          console.log("[DEBUG] Processing message:", message);

          if (message === '<<DONE>>') {
            console.log("[DEBUG] Received DONE signal");
            continue;
          }

          try {
            const match = message.match(/<step><step_name>(\w+)<\/step_name>(.*?)<\/step>/);
            if (match) {
              const [, stepName, content] = match;
              console.log("[DEBUG] Matched step:", stepName, "Content:", content);

              try {
                const data = JSON.parse(content);
                console.log("[DEBUG] Parsed data:", data);

                switch (stepName) {
                  case 'content':
                    accumulatedContent += data.content;
                    // Update the output with accumulated content
                    setOutputs(prev => {
                      const newOutputs = [...prev];
                      const lastIndex = newOutputs.length - 1;
                      if (lastIndex >= 0) {
                        newOutputs[lastIndex] = {
                          ...newOutputs[lastIndex],
                          result: {
                            answer: accumulatedContent,
                            tools_used: []
                          }
                        };
                      }
                      return newOutputs;
                    });
                    break;
                  case 'tool_start':
                    setOutputs(prev => {
                      const newOutputs = [...prev];
                      const lastIndex = newOutputs.length - 1;
                      if (lastIndex >= 0) {
                        const currentSteps = newOutputs[lastIndex].steps || [];
                        newOutputs[lastIndex] = {
                          ...newOutputs[lastIndex],
                          steps: [...currentSteps, {
                            name: data.tool,
                            result: "Running..."
                          }]
                        };
                      }
                      return newOutputs;
                    });
                    break;
                  case 'tool_result':
                    setOutputs(prev => {
                      const newOutputs = [...prev];
                      const lastIndex = newOutputs.length - 1;
                      if (lastIndex >= 0) {
                        const currentSteps = newOutputs[lastIndex].steps || [];
                        const updatedSteps = [...currentSteps];
                        const lastStepIndex = updatedSteps.length - 1;
                        if (lastStepIndex >= 0 && updatedSteps[lastStepIndex].name === data.tool) {
                          // Format the result for display
                          let formattedResult = data.result;
                          if (typeof data.result === 'string') {
                            try {
                              // Try to parse if it's a JSON string
                              formattedResult = JSON.parse(data.result);
                            } catch (e) {
                              // If not JSON, use as is
                              formattedResult = data.result;
                            }
                          }
                          updatedSteps[lastStepIndex] = {
                            ...updatedSteps[lastStepIndex],
                            result: formattedResult
                          };
                        }
                        newOutputs[lastIndex] = {
                          ...newOutputs[lastIndex],
                          steps: updatedSteps
                        };
                      }
                      return newOutputs;
                    });
                    break;
                  case 'final_answer':
                    if (data.answer) {
                      setOutputs(prev => {
                        const newOutputs = [...prev];
                        const lastIndex = newOutputs.length - 1;
                        if (lastIndex >= 0) {
                          newOutputs[lastIndex] = {
                            ...newOutputs[lastIndex],
                            result: { 
                              answer: data.answer,
                              tools_used: data.tools_used || []
                            }
                          };
                        }
                        return newOutputs;
                      });
                    }
                    break;
                  default:
                    console.log("[DEBUG] Unmatched step:", stepName, data);
                }
              } catch (parseError) {
                console.error("[DEBUG] Error parsing JSON:", parseError);
                console.log("[DEBUG] Failed content:", content);
              }
            } else {
              console.log("[DEBUG] No step match for message:", message);
            }
          } catch (error) {
            console.error("[DEBUG] Error processing message:", error);
            console.log("[DEBUG] Failed message:", message);
          }
        }
      }
    } catch (error) {
      console.error("[DEBUG] Error in handleSubmit:", error);
      setOutputs(prev => {
        const newOutputs = [...prev];
        const lastIndex = newOutputs.length - 1;
        if (lastIndex >= 0) {
          newOutputs[lastIndex] = {
            ...newOutputs[lastIndex],
            result: { 
              answer: "Error: " + (error instanceof Error ? error.message : "An error occurred"),
              tools_used: [],
              error: true
            }
          };
        }
        return newOutputs;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  // Submit form when Enter is pressed (without Shift)
  function submitOnEnter(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.code === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  }

  // Dynamically adjust textarea height based on content
  const adjustHeight = () => {
    const textArea = textAreaRef.current;
    if (textArea) {
      textArea.style.height = "auto";
      textArea.style.height = `${textArea.scrollHeight}px`;
    }
  };

  // Adjust height whenever text content changes
  useEffect(() => {
    adjustHeight();
  }, [text]);

  // Add resize event listener to adjust height on window resize
  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <form
      onSubmit={handleSubmit}
      className={`flex gap-3 z-10 ${
        outputs.length > 0 ? "fixed bottom-0 left-0 right-0 container pb-5" : ""
      }`}
    >
      <div className="w-full flex items-center bg-gray-800 rounded border border-gray-600">
        <textarea
          ref={textAreaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => submitOnEnter(e)}
          rows={1}
          className="w-full p-3 bg-transparent min-h-20 focus:outline-none resize-none"
          placeholder="Ask a question..."
        />

        <button
          type="submit"
          disabled={isGenerating || !text}
          className="disabled:bg-gray-500 bg-[#09BDE1] transition-colors w-9 h-9 rounded-full shrink-0 flex items-center justify-center mr-2"
        >
          <ArrowIcon />
        </button>
      </div>
    </form>
  );
};

const ArrowIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="lucide lucide-arrow-right"
  >
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);

export default TextArea;
