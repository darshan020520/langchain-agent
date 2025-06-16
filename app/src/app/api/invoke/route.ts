import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const { searchParams } = new URL(request.url);
  const content = searchParams.get('content');

  if (!content) {
    return NextResponse.json({ error: 'Content is required' }, { status: 400 });
  }

  try {
    const response = await fetch('http://localhost:8000/invoke', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get the response as a stream
    const stream = response.body;
    if (!stream) {
      throw new Error('No stream available');
    }

    // Return the stream directly
    return new NextResponse(stream);
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Failed to get response from backend' },
      { status: 500 }
    );
  }
} 