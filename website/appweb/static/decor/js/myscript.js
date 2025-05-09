$('#slider1, #slider2, #slider3').owlCarousel({
    loop: true,
    margin: 20,
    responsiveClass: true,
    responsive: {
        0: {
            items: 2,
            nav: false,
            autoplay: true,
        },
        600: {
            items: 4,
            nav: true,
            autoplay: true,
        },
        1000: {
            items: 6,
            nav: true,
            loop: true,
            autoplay: true,
        }
    }
})

$('.plus-cart').click(function(){
    var id=$(this).attr("pid").toString();
    var eml=this.parentNode.children[2] 
    $.ajax({
        type:"GET",
        url:"/pluscart",
        data:{
            prod_id:id
        },
        success:function(data){
            eml.innerText=data.quantity 
            document.getElementById("amount").innerText=data.amount 
            document.getElementById("totalamount").innerText=data.totalamount
        }
    })
})

$('.minus-cart').click(function(){
    var id=$(this).attr("pid").toString();
    var eml=this.parentNode.children[2] 
    $.ajax({
        type:"GET",
        url:"/minuscart",
        data:{
            prod_id:id
        },
        success:function(data){
            eml.innerText=data.quantity 
            document.getElementById("amount").innerText=data.amount 
            document.getElementById("totalamount").innerText=data.totalamount
        }
    })
})


$('.remove-cart').click(function(){
    var id=$(this).attr("pid").toString();
    var eml=this
    $.ajax({
        type:"GET",
        url:"/removecart",
        data:{
            prod_id:id
        },
        success:function(data){
            document.getElementById("amount").innerText=data.amount 
            document.getElementById("totalamount").innerText=data.totalamount
            eml.parentNode.parentNode.parentNode.parentNode.remove() 
        }
    })
})


$('.plus-wishlist').click(function(){
    var id=$(this).attr("pid").toString();
    $.ajax({
        type:"GET",
        url:"/pluswishlist",
        data:{
            prod_id:id
        },
        success:function(data){
            //alert(data.message)
            window.location.href = `http://localhost:8000/product-detail/${id}`
        }
    })
})


$('.minus-wishlist').click(function(){
    var id=$(this).attr("pid").toString();
    $.ajax({
        type:"GET",
        url:"/minuswishlist",
        data:{
            prod_id:id
        },
        success:function(data){
            window.location.href = `http://localhost:8000/product-detail/${id}`
        }
    })
})

$(document).ready(function(){
    $('.owl-carousel').owlCarousel({
      loop: true, // Lặp lại các slide
      margin: 10, // Khoảng cách giữa các slide
      nav: true, // Hiển thị nút điều hướng (prev/next)
      autoplay: true, // Tự động chạy
      autoplayTimeout: 5000, // Chuyển slide sau 5 giây
      autoplaySpeed: 1000, // Hiệu ứng chuyển đổi mất 1 giây
      autoplayHoverPause: true, // Tạm dừng khi di chuột
      items: 1 // Hiển thị 1 slide tại một thời điểm
      //animateOut: 'fadeOut', // Hiệu ứng khi slide rời đi
      //animateIn: 'fadeIn' // Hiệu ứng khi slide xuất hiện
    });
  });

$(document).ready(function(){
    // Khởi tạo Bootstrap Carousel
    $('#carouselExampleDark').carousel({
      interval: 5000, // Tất cả slide hiển thị 5 giây
      pause: 'hover', // Tạm dừng khi di chuột
      wrap: true // Lặp lại carousel
    });
});