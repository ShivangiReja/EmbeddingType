using System.Numerics;
using System.Text;
using System.Text.Json;

namespace EmbeddingType
{
    [TestFixture]
    internal class EmbeddingTypeTests
    {
        [Test]
        public void jsonArrayToFloat()
        {
            float[] expectedVectors = { -0.0026168018f, -0.024089903f, 0.03355637f };

            string jsonArr = JsonSerializer.Serialize(expectedVectors);
            EmbeddingVector jsonVector = EmbeddingVector.FromJson(Encoding.UTF8.GetBytes(jsonArr));
            EmbeddingVector<float> floats = jsonVector.To<float>();

            bool result = floats.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void jsonArrayToHalf()
        {
            Half[] expectedVectors = { (Half)(- 0.0026168018f), (Half)(- 0.024089903f), (Half)0.03355637f };

            string jsonArr = JsonSerializer.Serialize(expectedVectors);
            EmbeddingVector jsonVector = EmbeddingVector.FromJson(Encoding.UTF8.GetBytes(jsonArr));
            EmbeddingVector<Half> halfs = jsonVector.To<Half>();

            bool result = halfs.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);

        }

        [Test]
        public void jsonArrayToByte()
        {
            byte[] expectedVectors = { 1, 2, 3 };

            string jsonArr = JsonSerializer.Serialize(expectedVectors);
            EmbeddingVector jsonVector = EmbeddingVector.FromJson(Encoding.UTF8.GetBytes(jsonArr));
            EmbeddingVector<byte> bytes = jsonVector.To<byte>();

            bool result = bytes.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void jsonArrayToSByte()
        {
            sbyte[] expectedVectors = { 1, 2, 3 };

            string jsonArr = JsonSerializer.Serialize(expectedVectors);
            EmbeddingVector jsonVector = EmbeddingVector.FromJson(Encoding.UTF8.GetBytes(jsonArr));
            EmbeddingVector<sbyte> sbytes = jsonVector.To<sbyte>();

            bool result = sbytes.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }


        [Test]
        public void Base64StringToFloat()
        {
            float[] expectedVectors = { -0.0026168018f, -0.024089903f, 0.03355637f };

            string base64String = Convert.ToBase64String(expectedVectors.SelectMany(BitConverter.GetBytes).ToArray());
            EmbeddingVector base64Vector = EmbeddingVector.FromBase64(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<float> floats = base64Vector.To<float>();

            bool result = floats.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void Base64StringToHalf()
        {
            Half[] expectedVectors = { (Half)(-0.0026168018f), (Half)(-0.024089903f), (Half)0.03355637f };

            string base64String = Convert.ToBase64String(expectedVectors.SelectMany(BitConverter.GetBytes).ToArray());
            EmbeddingVector base64Vector = EmbeddingVector.FromBase64(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<Half> halfs = base64Vector.To<Half>();

            bool result = halfs.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void Base64StringToByte()
        {
            byte[] expectedVectors = { 1, 2, 3 };

            string base64String = Convert.ToBase64String(expectedVectors);
            EmbeddingVector base64Vector = EmbeddingVector.FromBase64(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<byte> bytes = base64Vector.To<byte>();

            bool result = bytes.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void Base64StringToSByte()
        {
            sbyte[] expectedVectors = { 1, 2, 3 };

            string base64String = Convert.ToBase64String(expectedVectors.Select(b => (byte)b).ToArray());
            EmbeddingVector base64Vector = EmbeddingVector.FromBase64(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<sbyte> sbytes = base64Vector.To<sbyte>();

            bool result = sbytes.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }
    }
}
