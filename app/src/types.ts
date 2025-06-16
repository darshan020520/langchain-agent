export type Step = {
  name: string;
  result: any;  // Allow any type for result since it varies by tool
};

export type ChatOutput = {
  question: string;
  steps: Step[];
  result: {
    answer: string;
    tools_used: string[];
  } | null;  // Make result nullable since it starts as null
};
