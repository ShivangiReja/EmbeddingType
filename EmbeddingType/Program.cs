using System;
using System.Numerics;
using System.Text.Json;
using System.Text;
using OpenAI.Embeddings;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Models;
using System.ClientModel;
using EmbeddingType;

namespace EmbeddingType
{
    class Program
    {
        public static void Main(string[] args)
        {
            Program p = new();
            p.OpenAIGetEmbedding();
        }

        public void VectorSearch()
        {
            Uri endpoint = new(Environment.GetEnvironmentVariable("SEARCH_ENDPOINT"));
            string key = Environment.GetEnvironmentVariable("SEARCH_API_KEY");
            AzureKeyCredential credential = new(key);
            string indexName = "mysearchindex";

            SearchClient searchClient = new SearchClient(endpoint, indexName, credential);
            ReadOnlyMemory<float> vectorizedResult = VectorSearchEmbeddings.SearchVectorizeDescription; // "Top hotels in town"

            Response<SearchResults<Hotel>> response = searchClient.Search<Hotel>(
                    new SearchOptions
                    {
                        VectorSearch = new()
                        {
                            Queries = { new VectorizedQuery(vectorizedResult) { KNearestNeighborsCount = 3, Fields = { "DescriptionVector" } } }
                        }
                    });

            // Get the first raw document of the search result and deserialize the vector property.
            using JsonDocument jsonDoc = JsonDocument.Parse(response.GetRawResponse().Content.ToStream());
            JsonElement doc = jsonDoc.RootElement.EnumerateObject().First().Value.EnumerateArray().First();

            if (doc.TryGetProperty("DescriptionVector", out JsonElement descriptionVector))
            {
                EmbeddingVector vector = EmbeddingVector.FromJson(Encoding.UTF8.GetBytes(descriptionVector.GetRawText()));
                EmbeddingVector<float> floats = vector.To<float>();

                // Print the elements
                foreach (float scalar in floats.Scalars.Span)
                {
                    Console.WriteLine(scalar);
                }
            }
        }

        public void OpenAIGetEmbedding()
        {
            ApiKeyCredential key = new("KEY");
            EmbeddingClient client = new("text-embedding-3-small", key);

            string description =
                "Best hotel in town if you like luxury hotels. They have an amazing infinity pool, a spa,"
                + " and a really helpful concierge. The location is perfect -- right downtown, close to all"
                + " the tourist attractions. We highly recommend this hotel.";

            Embedding embedding = client.GenerateEmbedding(description);
            ReadOnlyMemory<float> vector = embedding.Vector;

            Console.WriteLine($"Dimension: {vector.Length}");
            Console.WriteLine($"Floats: ");
            for (int i = 0; i < vector.Length; i++)
            {
                Console.WriteLine($"  [{i}] = {vector.Span[i]}");
            }
        }
    }
}

