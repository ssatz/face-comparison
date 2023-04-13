var FaceData  = {
                    'SourceImage': {'Bytes': ''},
                    'TargetImage': {'Bytes': ''},
                    'SimilarityThreshold':70
                 };

$(function() {
    $(document).on("change","#source-face-picker", function()
    {
        var files = !!this.files ? this.files : [];
        if (!files.length || !window.FileReader) return; // no file selected, or no FileReader support

        if (/^image/.test( files[0].type)){ // only image file
            var reader = new FileReader(); // instance of the FileReader
            reader.readAsDataURL(files[0]); // read the local file

            reader.onloadend = function(){ // set image data as background of div
              //Set the Base64 string return from FileReader as source.
              FaceData['SourceImage']['Bytes'] = reader.result.split('base64,')[1];

              $('.source-face-imagePreview').css("background-image", "url("+this.result+")");
            }
        }

    });
});

$(function() {
    $(document).on("change","#target-face-picker", function()
    {
        var files = !!this.files ? this.files : [];
        if (!files.length || !window.FileReader) return; // no file selected, or no FileReader support

        if (/^image/.test( files[0].type)){ // only image file
            var reader = new FileReader(); // instance of the FileReader
            reader.readAsDataURL(files[0]); // read the local file

            reader.onloadend = function(){ // set image data as background of div
              //Set the Base64 string return from FileReader as source.
               FaceData['TargetImage']['Bytes'] = reader.result.split('base64,')[1];

              $('.target-face-imagePreview').css("background-image", "url("+this.result+")");
            }
        }

    });
});

$("#upload-button").on("click", function(e) {
    if (['SourceImage']['Bytes'] === "") {
        alert("Please select source face.");
        return;
    }

    if (FaceData['TargetImage']['Bytes'] === "") {
        alert("Please select target face.");
        return;
    }

    var UPLOAD_URL = "/api/upload";

    $.ajax({
        url: UPLOAD_URL,
        method: "POST",
        data: FaceData,
        success: function(data) {
            data = JSON.parse(data);
            console.log(data);
        }
      });
});

