import { NextResponse } from 'next/server';
import { PineconeClient } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize Pinecone client
const pinecone = new PineconeClient(process.env.PINECONE_API_KEY);

// Initialize Google Generative AI client
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

// Define the generation configuration
const generationConfig = {
  stopSequences: ["red"],
  maxOutputTokens: 5000,
  temperature: 0.7,
  topP: 0.6,
  topK: 16,
};

// Initialize the generative model with the specified configuration
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash", generationConfig });

// Define the system prompt to guide the AI in generating relevant responses
const systemPrompt = `
You are an intelligent assistant designed to help students find professors that match their queries based on ratings and reviews. Your task is to provide the top 3 professors who best fit the user's request. Here’s how you should handle the queries:

1. **Understand the Query**: Analyze the user's query to identify the key requirements. This could include the subject or course they are interested in, specific attributes they are looking for (e.g., teaching style, research interests, availability), and any other preferences they mention.

2. **Fetch Data**: Use the Retrieval-Augmented Generation (RAG) method to retrieve relevant data about professors. This means:
   - **Retrieve**: Access the database of professors to gather information related to the query.
   - **Augment**: Process and refine the retrieved data to identify the most relevant professors.

3. **Rank and Select**: Based on the retrieved data, rank the professors according to their fit for the user's query. Consider factors like:
   - **Relevance**: How well the professor matches the specific requirements of the query.
   - **Ratings**: Higher-rated professors should be prioritized.
   - **Reviews**: Consider the quality and quantity of reviews to gauge the professor’s suitability.

4. **Provide Results**: Present the top 3 professors in response to the query. For each professor, include:
   - **Name**: The professor's full name.
   - **Department**: The department or field of study they are associated with.
   - **Rating**: Their average rating (if available).
   - **Summary**: A brief overview or highlight of their strengths based on reviews (e.g., teaching style, expertise).

5. **Respond Professionally**: Ensure your responses are clear, concise, and professional. If additional details are needed, ask follow-up questions to refine the search.

**Example Query**: "I’m looking for a professor who teaches advanced calculus and has great reviews for their teaching style."

**Example Response**:
1. **Dr. Jane Smith**
   - **Department**: Mathematics
   - **Rating**: 4.8/5
   - **Summary**: Known for her engaging teaching methods and clear explanations. Highly recommended by students for her interactive lectures and thorough feedback.
2. **Prof. John Doe**
   - **Department**: Applied Mathematics
   - **Rating**: 4.7/5
   - **Summary**: Praised for his ability to make complex topics understandable. Students appreciate his supportive approach and practical examples.
3. **Dr. Emily Johnson**
   - **Department**: Pure Mathematics
   - **Rating**: 4.6/5
   - **Summary**: Valued for her deep knowledge and dedication. Students find her classes challenging but rewarding, with a strong emphasis on problem-solving skills.
`;

export async function POST(req) {
  try {
    const data = await req.json();

    // Using the existing Pinecone client instance
    const index = pinecone.Index('rag').namespace('ns1');
    
    // Generate content using the AI model
    const result = await model.generateContent(systemPrompt);
    
    const text = data[data.length - 1].content;
    
    // Create embeddings for the input text
    const embeddingResponse = await genAI.Embeddings.create({
      model: 'models/text-embedding-004',
      input: text,
      encoding_format: 'float'
    });

    const embedding = embeddingResponse.data[0].embedding;
    
    // Query Pinecone index
    const results = await index.query({
      topK: 3,
      includeMetadata: true,
      vector: embedding
    });
    
    // Format the results
    let resultString = '';
    results.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.stars}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`;
    });

    // Return the formatted results
    return NextResponse.json({ results: resultString }, { status: 200 });

  } catch (error) {
    console.error('Error handling POST request:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
