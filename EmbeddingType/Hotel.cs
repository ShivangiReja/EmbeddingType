using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EmbeddingType
{
    public class Hotel
    {
        public string HotelId { get; set; }
        public string HotelName { get; set; }
        public string Description { get; set; }
        public ReadOnlyMemory<float> DescriptionVector { get; set; }
        public string Category { get; set; }
        public ReadOnlyMemory<float> CategoryVector { get; set; }
    }
}
