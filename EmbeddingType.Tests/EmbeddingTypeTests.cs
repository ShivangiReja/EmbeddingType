using System.Buffers.Binary;
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
            EmbeddingVector base64Vector = new Base64Vector(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<float> floats = base64Vector.To<float>();

            bool result = floats.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void Base64StringToHalf()
        {
            Half[] expectedVectors = { (Half)(-0.0026168018f), (Half)(-0.024089903f), (Half)0.03355637f };

            string base64String = Convert.ToBase64String(expectedVectors.SelectMany(BitConverter.GetBytes).ToArray());
            EmbeddingVector base64Vector = new Base64Vector(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<Half> halfs = base64Vector.To<Half>();

            bool result = halfs.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void Base64StringToByte()
        {
            byte[] expectedVectors = { 1, 2, 3 };

            string base64String = Convert.ToBase64String(expectedVectors);
            EmbeddingVector base64Vector = new Base64Vector(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<byte> bytes = base64Vector.To<byte>();

            bool result = bytes.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        [Test]
        public void Base64StringToSByte()
        {
            sbyte[] expectedVectors = { 1, 2, 3 };

            string base64String = Convert.ToBase64String(expectedVectors.Select(b => (byte)b).ToArray());
            EmbeddingVector base64Vector = new Base64Vector(Encoding.UTF8.GetBytes(base64String));
            EmbeddingVector<sbyte> sbytes = base64Vector.To<sbyte>();

            bool result = sbytes.Scalars.Span.SequenceEqual(expectedVectors);
            Assert.IsTrue(result);
        }

        internal sealed class Base64Vector : EmbeddingVector
        {
            private readonly ReadOnlyMemory<byte> _utf8Base64;

            public Base64Vector(ReadOnlyMemory<byte> utf8Base64) => _utf8Base64 = utf8Base64;

            public override EmbeddingVector<T> To<T>()
            {
                try
                {
                    var decodedBytes = Convert.FromBase64String(Encoding.UTF8.GetString(_utf8Base64.Span));
                    var span = new ReadOnlySpan<byte>(decodedBytes);

                    T[] array;

                    if (typeof(T) == typeof(float))
                    {
                        array = new T[span.Length / sizeof(float)];
                        for (int i = 0; i < array.Length; i++)
                        {
                            float value = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i * sizeof(float), sizeof(float)));
                            array[i] = (T)(object)value;
                        }
                    }
#if NET5_0_OR_GREATER
                    else if (typeof(T) == typeof(Half))
                    {
                        array = new T[span.Length / sizeof(short)];
                        for (int i = 0; i < array.Length; i++)
                        {
                            Half value = (Half)BinaryPrimitives.ReadHalfLittleEndian(span.Slice(i * sizeof(short), sizeof(short)));
                            array[i] = (T)(object)value;
                        }
                    }
#endif
                    else if (typeof(T) == typeof(byte))
                    {
                        array = new T[span.Length];
                        for (int i = 0; i < array.Length; i++)
                        {
                            byte value = span[i];
                            array[i] = (T)(object)value;
                        }
                    }
                    else if (typeof(T) == typeof(sbyte))
                    {
                        array = new T[span.Length];
                        for (int i = 0; i < array.Length; i++)
                        {
                            sbyte value = (sbyte)span[i];
                            array[i] = (T)(object)value;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Type {typeof(T)} is not supported.");
                    }

                    return new EmbeddingVector<T>(array);
                }
                catch (Exception e)
                {
                    throw new NotSupportedException(e.Message, e);
                }
            }

            public override void Write(Stream stream, string format)
            {
                if (!format.Equals("J", StringComparison.Ordinal))
                {
                    throw new NotSupportedException();
                }

                using var writer = new Utf8JsonWriter(stream);
                writer.WriteStringValue(_utf8Base64.Span);
            }
        }

    }
}
