var dataArray = [
    "1st phase jp nagar",
        "5th phase jp nagar",
        "7th phase jp nagar",
        "8th phase jp nagar",
        "9th phase jp nagar",
        "aecs layout",
        "abbigere",
        "akshaya nagar",
        "ambalipura",
        "ambedkar nagar",
        "amruthahalli",
        "anandapura",
        "ardendale",
        "arekere",
        "attibele",
        "btm 2nd stage",
        "btm layout",
        "badavala nagar",
        "balagere",
        "banashankari",
        "banashankari stage iii",
        "bannerghatta road",
        "battarahalli",
        "begur",
        "begur road",
        "bellandur",
        "bharathi nagar",
        "bhoganhalli",
        "billekahalli",
        "binny pete",
        "bisuvanahalli",
        "bommanahalli",
        "bommasandra",
        "bommenahalli",
        "brookefield",
        "budigere",
        "cv raman nagar",
        "chandapura",
        "channasandra",
        "chikka tirupathi",
        "choodasandra",
        "dairy circle",
        "dasanapura",
        "dasarahalli",
        "devanahalli",
        "dodda nekkundi",
        "doddathoguru",
        "domlur",
        "epip zone",
        "electronic city",
        "electronic city phase ii",
        "electronics city phase 1",
        "frazer town",
        "gollarapalya hosahalli",
        "gottigere",
        "green glen layout",
        "gubbalala",
        "gunjur",
        "hsr layout",
        "haralur road",
        "harlur",
        "hebbal",
        "hebbal kempapura",
        "hegde nagar",
        "hennur",
        "hennur road",
        "hoodi",
        "horamavu agara",
        "hormavu",
        "hosa road",
        "hosakerehalli",
        "hoskote",
        "hosur road",
        "hulimavu",
        "iblur village",
        "indira nagar",
        "jp nagar",
        "jakkur",
        "jalahalli",
        "jalahalli east",
        "jigani",
        "kr puram",
        "kadugodi",
        "kaggadasapura",
        "kaggalipura",
        "kalena agrahara",
        "kalyan nagar",
        "kambipura",
        "kammasandra",
        "kanakapura",
        "kanakpura road",
        "kannamangala",
        "karuna nagar",
        "kasavanhalli",
        "kenchenahalli",
        "kengeri",
        "kengeri satellite town",
        "kereguddadahalli",
        "kodichikkanahalli",
        "kodihalli",
        "koramangala",
        "kothanur",
        "kudlu",
        "kudlu gate",
        "kundalahalli",
        "lakshminarayana pura",
        "lingadheeranahalli",
        "magadi road",
        "mahadevpura",
        "mallasandra",
        "malleshwaram",
        "marathahalli",
        "marsur",
        "munnekollal",
        "mysore road",
        "nagarbhavi",
        "nagavarapalya",
        "neeladri nagar",
        "old airport road",
        "old madras road",
        "padmanabhanagar",
        "panathur",
        "parappana agrahara",
        "pattandur agrahara",
        "prithvi layout",
        "rachenahalli",
        "raja rajeshwari nagar",
        "rajaji nagar",
        "rajiv nagar",
        "ramagondanahalli",
        "ramamurthy nagar",
        "rayasandra",
        "sahakara nagar",
        "sarjapur",
        "sarjapur  road",
        "sector 2 hsr layout",
        "seegehalli",
        "somasundara palya",
        "sonnenahalli",
        "subramanyapura",
        "talaghattapura",
        "thanisandra",
        "thigalarapalya",
        "thubarahalli",
        "tumkur road",
        "uttarahalli",
        "varthur",
        "vidyaranyapura",
        "vijayanagar",
        "vittasandra",
        "whitefield",
        "yelahanka",
        "yelahanka new town",
        "yeshwanthpur"
    ];

function populateDropdown() {
    var select = document.getElementById("location");

    select.innerHTML = "";

    for (var index = 0; index < dataArray.length; index++) {
        var option = document.createElement("option");
        option.text = dataArray[index];
        select.appendChild(option);
    };
}
    
window.onload = function() {
    populateDropdown();
};

function pred(){
    var squareFeet = document.getElementById("square_feet").value;

    var location = document.getElementById("location").value;

    var bhk = document.querySelector('input[name="bhk"]:checked').value;

    var url="http://127.0.0.1:5000/predict_home_price";
    $.post(url, {
        location: location,
        total_sqft: squareFeet,
        bhk: bhk,
    },function(data, status) {
        console.log(data.estimated_price);
        document.getElementById('pred').innerHTML = data.estimated_price + " lakhs";
        document.getElementById("pred").style.display="block";
        console.log(status);
    });
  }
  
  
